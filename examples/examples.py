# matplotalt also works outside the notebook environment to generate alt text
import os
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from matplotalt import show_with_alt, show_with_api_alt, generate_alt_text, infer_chart_type


if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    # Seattle bikes vs sunshine each month
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec"]
    BIKES_OVER_FREMONT_BRIDGE = np.array([112252.8, 103497.2, 136189.2, 165020.4, 231792, 221274.8, 234421.6, 224087.2, 190238, 166078, 121548, 89695.6])
    SUNSHINE_HOURS = np.array([69, 108, 178, 207, 253, 268, 312, 281, 221, 142, 72, 52])

    def line_sun():
        avg_bikes = np.mean(BIKES_OVER_FREMONT_BRIDGE)
        avg_hours = np.mean(SUNSHINE_HOURS)
        normalized_bikes = 100 * (BIKES_OVER_FREMONT_BRIDGE - avg_bikes) / avg_bikes
        normalized_sunshine = 100 * (SUNSHINE_HOURS - avg_hours) / avg_hours

        plt.title("Average Monthly Hours of Sunshine in Seattle vs. \
                \nNumber of Bikes that Cross Fremont Bridge")
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.25)
        plt.plot(normalized_bikes, ls="--", label="# Bikes Crossing Fremont Bridge", alpha=0.75)
        plt.plot(normalized_sunshine, label="Hours of Sunshine", alpha=0.75)
        plt.xticks(ticks=list(range(0, 11, 2)), labels=MONTHS[::2])
        plt.xlabel("Month")
        plt.ylabel("Change from Yearly Average (%)")
        plt.annotate('234421 bikes in July', xy=(6, 40), xytext=(8, 40), arrowprops=dict(facecolor='black', shrink=0.05, alpha=0.5))
        plt.legend()


    def bar_sun():
        fig, ax = plt.subplots()
        ax.set_title("Average Number of Bikes Crossing Fremont Bridge Each Month from 2014-2018")
        ax.barh(list(range(12)), BIKES_OVER_FREMONT_BRIDGE)
        ax.set_xlabel("Avg. # Bikes Crossing Fremont Bridge")
        ax.set_ylabel("Month")
        ax.set_yticks(ticks=list(range(0, 12)), labels=MONTHS)
        plt.tight_layout()


    def pie_sun():
        colors = plt.cm.tab20(list(range(12)))
        plt.pie(SUNSHINE_HOURS, labels=MONTHS, autopct='%1.1f%%', pctdistance=1.25,
                                labeldistance=0.75, colors=colors, textprops={'fontsize': 9})
        plt.title("Percentage of Annual Sunshine")

    # show_with_alt and show_with_api_alt work the same in non-notebook environments, with the exception
    # that only "img_file" and "txt_file" display methods are supported.
    line_sun()
    line_alt_text = show_with_alt(desc_level=3, methods="img_file", return_alt=True)
    print(f"Line chart alt text: {line_alt_text}")

    bar_sun()
    bar_alt_text = show_with_api_alt(OPENAI_API_KEY, return_alt=True)
    print(f"Bar chart alt text: {bar_alt_text}")

    # Note that an interactive window will not pop up with show_with_alt and show_with_api_alt methods.
    # To preserve matplotlib's default behavior, first use generate_alt_text and then call plt.show()
    pie_sun()
    pie_alt_text = generate_alt_text()
    plt.show()
    print(f"Pie chart alt text: {pie_alt_text}")