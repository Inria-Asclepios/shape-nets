import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def main():
    res = 'Datasets/Shapes/Human_pose/results.csv'
    df = pd.read_csv(res, delimiter=';')
    new_df = pd.melt(df, id_vars=["noise"])
    new_df = new_df.rename({'noise': 'Noise (%)', 'value': 'Test accuracy', 'variable': 'Input feature'}, axis='columns')

    sns.set_theme()
    sns.set_context("talk")
    palette = sns.color_palette("mako_r", 3)

    sns.lineplot(x="Noise (%)", y="Test accuracy", data=new_df, hue="Input feature", style="Input feature",
                 palette=palette)
    plt.show()

    breakpoint()


if __name__ == '__main__':
    main()
