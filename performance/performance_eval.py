import pandas as pd
import matplotlib.pyplot as plt

# Sort models and print head and tail
df = pd.read_csv("performance.csv")
df = df.sort_values(by=['Loss'], ascending=[True])
print(df[0:25])

# Visualise model performance
unique_opt = df.Optimizer.unique()
for metric in ['Accuracy', 'Loss', 'Time']:
    plt.figure()
    for opt in unique_opt:
        sub = df[df['Optimizer'] == opt]
        sub = sub.sort_values(by='Activation', ascending=True)
        plt.plot(sub['Activation'], sub[metric], 'o-', label=opt)

    plt.title(f"{metric} for Different Models")
    plt.xlabel("Activation Function")
    plt.ylabel(f"{metric}")
    plt.legend()
    plt.savefig(f"{metric}.png")
plt.show()

