import csv
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\pyformaster\pyfordata\Ancillary_Data\fusion\comparison1.csv")
print(df.head())

# with plt.style.context("ggplot"):
#
#     plt.figure(1)
#     ax = plt.subplot(211)
#
#     plt.scatter(df['Plot'], df['Z Maximum'])
#     plt.axhline(0, color='gray')
#     plt.ylim(-10,10)
#     plt.xlim(0,20.5)
#
#
#     plt.subplot(212)
#     plt.scatter(df['Plot'], df['Z P95'])
#     plt.axhline(0, color='gray')
#     plt.xlim(0,20.5)
# plt.show()

def plot_all():
    cols = df.columns.values[1:13] # All fields except plot
    with plt.style.context("ggplot"):
        i = 0
        plt.figure(1)
        plt.suptitle("FUSION Pyfor Difference Comparison", fontsize = 20)
        for column in cols:

            ax = plt.subplot(6,2,i+1)
            ax.set_title(column)
            ax.title.set_fontsize(16)


            plt.scatter(df['Plot'], df[column])
            plt.axhline(0, color='gray')

            #Find largest of the two y ranges, and set the y range to that. Centers y over 0
            y_raw = ax.get_ylim()
            abs_raw = [abs(lim) for lim in y_raw]
            y_new = max(abs_raw)
            ax.set_ylim(-y_new, y_new)
            ax.set_xlim(0,21)

            ax.tick_params(axis = 'both', which='major', labelsize = 10)
            ax.tick_params(axis='both', which='minor', labelsize=10)

            i+=1
    plt.show()


plot_all()