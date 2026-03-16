import random
from token_scheduler import TokenLRScheduler


scheduler = TokenLRScheduler(
    base_lr=3e-4,
    min_lr=3e-5,
    warmup_tokens=1e8,
    total_tokens=3e9,
)

tokens = []
lrs = []

total_tokens = 0

for step in range(2000):

    tokens_batch = random.randint(20000, 80000)

    lr = scheduler.update(tokens_batch)

    total_tokens += tokens_batch

    tokens.append(total_tokens)
    lrs.append(lr)
