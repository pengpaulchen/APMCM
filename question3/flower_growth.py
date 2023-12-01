import os
import matplotlib.pyplot as plt


def get_maturity_counts(dir_path):
    maturity_counts = {0: 0, 1: 0}
    file_list = os.listdir(dir_path)
    for file_name in file_list:
        if file_name.endswith(".txt"):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    maturity = int(line.strip().split()[0])
                    if maturity in maturity_counts:
                        maturity_counts[maturity] += 1
    return maturity_counts


def plot_maturity_counts(maturity_counts):
    maturity_levels = [ 'unflowering','flowering']
    counts = [maturity_counts[0], maturity_counts[1]]
    plt.bar(maturity_levels, counts)
    plt.xlabel('Maturity Level')
    plt.ylabel('Count')
    plt.title('Maturity Distribution of All Apples')
    plt.show()


dir_path = r"./flower"  # 更改为你的txt文件路径
maturity_counts = get_maturity_counts(dir_path)
plot_maturity_counts(maturity_counts)



