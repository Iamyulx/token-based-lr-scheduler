import math
import time


class TokenLRScheduler:
    """
    Token-based learning rate scheduler for LLM training.

    Features
    --------
    • Token-based warmup
    • Cosine decay based on total token budget
    • Adaptive throughput tracking
    """

    def __init__(
        self,
        base_lr,
        min_lr,
        warmup_tokens,
        total_tokens,
        throughput_smoothing=0.9,
    ):

        self.base_lr = base_lr
        self.min_lr = min_lr

        self.warmup_tokens = warmup_tokens
        self.total_tokens = total_tokens

        self.tokens_processed = 0

        self.last_time = time.time()
        self.last_tokens = 0

        self.throughput_smoothing = throughput_smoothing
        self.smoothed_throughput = None

    def update(self, tokens_in_batch):

        self.tokens_processed += tokens_in_batch

        lr = self.compute_lr()

        self.update_throughput()

        return lr

    def compute_lr(self):

        t = self.tokens_processed

        if t < self.warmup_tokens:

            return self.base_lr * t / self.warmup_tokens

        progress = (t - self.warmup_tokens) / (
            self.total_tokens - self.warmup_tokens
        )

        progress = min(progress, 1.0)

        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

        return lr

    def update_throughput(self):

        current_time = time.time()
        delta_time = current_time - self.last_time

        if delta_time == 0:
            return

        tokens_since_last = self.tokens_processed - self.last_tokens

        throughput = tokens_since_last / delta_time

        if self.smoothed_throughput is None:

            self.smoothed_throughput = throughput

        else:

            self.smoothed_throughput = (
                self.throughput_smoothing * self.smoothed_throughput
                + (1 - self.throughput_smoothing) * throughput
            )

        self.last_time = current_time
        self.last_tokens = self.tokens_processed

    def get_state(self):

        return {
            "tokens_processed": self.tokens_processed,
            "throughput": self.smoothed_throughput,
        }
