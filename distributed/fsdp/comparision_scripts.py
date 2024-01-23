class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            *(nn.Linear(10000, 10000) for _ in range(10))
        )

    def forward(self, x):
        return self.linear(x)


def test_fp32():
    model = Layer().cuda()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(10000).cuda()
    for i in range(10):
        optimizer.zero_grad()
        output = model(data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        memory = max_memory_allocated()
        print(f'step memory allocate: {memory / 1e9:.3f}G')


def test_fp16():
    torch.cuda.init()
    model = Layer().cuda()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(10000).cuda()
    for _ in range(10):
        with autocast(device_type='cuda'):
            optimizer.zero_grad()
            output = model(data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        memory = max_memory_allocated()
        print(f'memory allocated: {memory / 1e9:.3f}G')

# To save this part of the parameters, you need to pass cache_enabled=False to autocast.
def test_fp16():
    torch.cuda.init()
    model = Layer().cuda()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(10000).cuda()
    for _ in range(10):
        with autocast(device_type='cuda', cache_enabled=False):
            optimizer.zero_grad()
            output = model(data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        memory = max_memory_allocated()
        print(f'memory allocated: {memory / 1e9:.3f}G')


# DDp Training
def _test_ddp_fp16():
    rank = dist.get_rank()
    model = DistributedDataParallel(Layer().cuda())
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(10000).cuda()
    for _ in range(10):
        with autocast(device_type='cuda', cache_enabled=False):
            optimizer.zero_grad()
            output = model(data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        memory = max_memory_allocated()
        if rank == 0:
            print(f'memory allocated: {memory / 1e9:.3f}G')

# FSDP

from torch.distributed.fsdp.wrap import _module_wrap_policy

def _test_fsdp_fp16():
    rank = dist.get_rank()
    fsdp_model = FullyShardedDataParallel(
        module=Layer(), device_id=rank,
        auto_wrap_policy=partial(
            _module_wrap_policy,
            module_classes=nn.Linear))
    optimizer = SGD(fsdp_model.parameters(), lr=0.1, momentum=0.9)
    data = torch.ones(10000).cuda()
    for _ in range(10):
        optimizer.zero_grad()
        output = fsdp_model(data)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        memory = max_memory_allocated()
        if rank == 0:
            print(f'step memory allocate: {memory / 1e9:.3f}G')
        torch.cuda.reset_max_memory_allocated()
