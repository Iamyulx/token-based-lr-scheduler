import matplotlib.pyplot as plt
from simulate_training import tokens, lrs


plt.plot(tokens, lrs)

plt.xlabel("Tokens processed")
plt.ylabel("Learning Rate")

plt.title("Token-based LR Schedule")

plt.grid(True)

plt.savefig("token_lr_curve.png", dpi=300, bbox_inches="tight")

plt.show()
