import matplotlib.pyplot as plt
import json
import numpy as np
import torch

def main():
    with open("pleasure_data.json", "r") as f:
        data = json.load(f)


    etas = data["etas"][-1]
    lrs = data["lrs"]
    grads = data["grads"]
    grad_time_fracs = data["grad_time_fracs"]

    # print(grad_time_fracs)
    # Plotting grad_time_fracs

    # compute average time over blocks per forward
    avg_grad_time_fracs = [sum(l) / len(l) for l in grad_time_fracs]

    # plt.figure(figsize=(10, 6))
    # plt.subplot(2, 2, 1)
    # plt.plot(avg_grad_time_fracs, label='Grad Time Fracs')
    # plt.title('Grad Time Fracs Over Time')
    # plt.xlabel('Forward Step')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()


    # Plotting etas
    breakpoint()
    lrs_tensors = torch.tensor(etas)
    print(lrs_tensors.shape)

    # grad_norm_means = []
    # for gradlist in grads:
    #     # compute norm of each gradient vector
    #     grad_norms = [torch.norm(torch.tensor(g)).item() for g in gradlist]
    #     mean_grad_norm = np.mean(grad_norms)

    #     grad_norm_means.append(mean_grad_norm)

    # # Plotting gradient norms
    # plt.figure(figsize=(10, 6))
    # plt.subplot(2, 2, 2)
    # plt.plot(grad_norm_means, label='Gradient Norms')
    # plt.title('Gradient Norms Over Time')
    # plt.xlabel('Time Step')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()