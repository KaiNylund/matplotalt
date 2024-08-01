import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patheffects
from scipy.interpolate import pchip
from collections import OrderedDict

# Seattle bikes vs sunshine each month
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec"]
BIKES_OVER_FREMONT_BRIDGE = np.array([112252.8, 103497.2, 136189.2, 165020.4, 231792, 221274.8, 234421.6, 224087.2, 190238, 166078, 121548, 89695.6])
SUNSHINE_HOURS = np.array([69, 108, 178, 207, 253, 268, 312, 281, 221, 142, 72, 52])
# Random 2d gaussians
BLUE_RANDOM_GAUSSIAN_2D = np.random.multivariate_normal([1, 2], [[4, -2], [-2, 3]], size=350)
ORANGE_RANDOM_GAUSSIAN_2D = np.random.multivariate_normal([-4, 3], [[2, 3], [-2, 4]], size=250)
# Anscombe's quartet
X_ANS = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
Y1_ANS = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
Y2_ANS = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
Y3_ANS = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
X4_ANS = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19])
Y4_ANS = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50])



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
    cmap = plt.cm.tab20
    colors = cmap(list(range(12)))
    plt.pie(SUNSHINE_HOURS, labels=MONTHS, autopct='%1.1f%%', pctdistance=1.25,
                            labeldistance=0.75, colors=colors, textprops={'fontsize': 9})
    plt.title("Percentage of Annual Sunshine")


def radial_line_sun():
    SUN_COLOR = "#FFB631"
    # Convert bike and sunshine stats into radial coordinates
    monthnums = np.array(list(range(12)))
    r = SUNSHINE_HOURS
    theta = (monthnums * np.pi) / 6
    interp = pchip(theta, r)
    tt = np.linspace(0, 2 * np.pi, 360)
    r = list(SUNSHINE_HOURS) + [SUNSHINE_HOURS[0]]
    theta = list(theta) + [theta[0]]
    interptt = list(interp(tt)) + [interp(tt)[0]]
    tt = list(tt) + [tt[0]]
    # Start building matplotlib figure
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    fig.suptitle("Avg. Monthly Hours of Sunshine in Seattle", fontsize=14)
    # Configure twin axes
    ax.set_rticks(np.linspace(0, np.amax(r), 5))  # Reduce the number of radial ticks
    bike_labels = (np.linspace(0, np.amax(BIKES_OVER_FREMONT_BRIDGE), 5).astype(int) // 1000).astype(str)
    bike_labels = [l + "K" for l in bike_labels]
    bike_labels[0] = "0"
    ax.set_rlabel_position(ax.get_rlabel_position())
    ax.set_rlabel_position(90)
    ax.tick_params(labelsize=13)
    # Add outside month labels
    ax.set_xticks(np.arange(0,2.0*np.pi,np.pi/6.0))
    ax.set_xticklabels(MONTHS, fontsize=15)
    # Add dashed grid lines
    ax.grid(True, alpha=0.5, linestyle="dashed")
    ax.set_axisbelow(True)
    plt.setp(ax.get_yticklabels(), color=SUN_COLOR)
    ax.set_ylim([0, 350])
    # Plot the radial lines and fills
    ax.plot(theta, r, linewidth=2, color=SUN_COLOR, label='Avg. Hours of Sunshine',
            solid_capstyle='round', zorder=100, marker="o", alpha=0.75)
    ax.fill_between(theta, r, facecolor=SUN_COLOR, alpha=0.1, zorder=99)
    ax.legend(loc="lower left", bbox_to_anchor=(.5 + np.cos(np.pi / 4)/2, .5 + np.sin(np.pi / 4)/2))
    # Add outlines for the tick labels to increase contrast
    for tick in (ax.get_yticklabels()):
        tick.set_path_effects([patheffects.withStroke(linewidth=1, foreground='black')])
    # Make sure ticks are drawn on top of lines and fills
    ax.tick_params(zorder=200)
    plt.gcf().set_size_inches(7, 7)
    plt.tight_layout()


def scatter_gaussian():
    plt.scatter(BLUE_RANDOM_GAUSSIAN_2D[:, 0], BLUE_RANDOM_GAUSSIAN_2D[:, 1], label="The blue dots")
    plt.scatter(ORANGE_RANDOM_GAUSSIAN_2D[:, 0], ORANGE_RANDOM_GAUSSIAN_2D[:, 1], label="The orange dots")
    plt.title("Points from 2d Gaussian Distributions")
    plt.xlabel("Random Gaussian x")
    plt.ylabel("Random Gaussian y")
    plt.legend()


def strip_gaussian():
    sns.stripplot([BLUE_RANDOM_GAUSSIAN_2D[:, 0], ORANGE_RANDOM_GAUSSIAN_2D[:, 0]],
              jitter=False, size=20, linewidth=1, alpha=0.05, orient="h")
    plt.gcf().set_size_inches(10, 3)
    plt.xlabel("Random Gaussian x")
    plt.yticks(ticks=[0, 1], labels=["The blue dots", "The orange dots"])
    plt.tight_layout()


def heatmap_gaussian():
    combined_points = np.concatenate((BLUE_RANDOM_GAUSSIAN_2D, ORANGE_RANDOM_GAUSSIAN_2D))
    plt.hist2d(combined_points[:, 0], combined_points[:, 1], bins=(range(-8, 8, 2), range(-4, 10, 2)))
    plt.title('Number of points from combined gaussians')
    plt.colorbar(label="Number of gaussian points")


def contour_gaussian():
    combined_points = np.concatenate((BLUE_RANDOM_GAUSSIAN_2D, ORANGE_RANDOM_GAUSSIAN_2D))
    hist_bins, xbins, ybins = np.histogram2d(combined_points[:, 0], combined_points[:, 1], bins=9)
    X, Y = np.meshgrid(xbins[:-1], ybins[:-1])

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, hist_bins)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Number of points from combined gaussians')


def subplots_anscombs():
    x_order = np.argsort(X_ANS)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle("Anscombe's Quartet")
    axs[0][0].plot(X_ANS[x_order], Y1_ANS[x_order], "-o")
    axs[0][1].plot(X_ANS[x_order], Y2_ANS[x_order], "-o")
    axs[1][0].plot(X_ANS[x_order], Y3_ANS[x_order], "-o")
    axs[1][1].plot(X4_ANS, Y4_ANS, "-o")


def boxplot_anscombs():
    bp = sns.boxplot([Y1_ANS, Y2_ANS, Y3_ANS, Y4_ANS])
    plt.ylabel("y Value in Anscombe's Quartet")
    plt.xlabel("Subplot Num")
    plt.title("Distributional differences in Anscombe's quartet")


CHART_FUNC_TO_DATA = {
    line_sun: {
        "type": "line",
        "data": OrderedDict([('x', [list(range(12)), list(range(12))]),
                             ('y', [np.array([-32.51655182, -37.7801896 , -18.12661403,  -0.79405046, 39.34728952,  33.02462388,  40.92813628,  34.71536522, 14.36611127,  -0.158249  , -26.92852063, -46.07735063]), np.array([-61.71983356, -40.08321775,  -1.2482663 ,  14.84049931, 40.36061026,  48.68238558,  73.09292649,  55.89459085, 22.6074896 , -21.22052705, -60.0554785 , -71.15117892])])]),
        "ticklabels": OrderedDict([('x', MONTHS[::2]),
                                   ('y', ['−80', '−60', '−40', '−20', '0', '20', '40', '60', '80', '100'])]),
        "axislabels": OrderedDict([('x', 'Month'), ('y', 'Change from Yearly Average (%)')]),
        "axistypes": OrderedDict([('x', 'datetime'), ('y', 'linear')]),
    },
    bar_sun: {
        "type": "bar",
        "data": OrderedDict([('x', [BIKES_OVER_FREMONT_BRIDGE]),
                             ('y', [list(range(12))])]),
        "ticklabels": OrderedDict([('x', ['0', '50000', '100000', '150000', '200000', '250000']),
                                   ('y', MONTHS)]),
        "axislabels": OrderedDict([('x', 'Avg. # Bikes Crossing Fremont Bridge'), ('y', 'Month')]),
        "axistypes": OrderedDict([('x', 'linear'), ('y', 'datetime')])
    },
    pie_sun: {
        "type": "pie",
        "data": OrderedDict([('x', [3.1900137662887573, 4.993065074086189, 8.229310810565948, 9.570041298866272, 11.696717888116837, 12.390198558568954, 14.424410462379456, 12.991215288639069, 10.217291116714478, 6.564956158399582, 3.328710049390793, 2.404068410396576])]),
        "ticklabels": OrderedDict([('x', MONTHS), ('y', [])]),
        "axislabels": OrderedDict([('x', ''), ('y', '')]),
        "axistypes": OrderedDict(),
    },
    radial_line_sun: {
        "type": "radial line",
        "data": OrderedDict([('x', [np.array([0, 0.52359878, 1.04719755, 1.57079633, 2.0943951, 2.61799388, 3.14159265, 3.66519143, 4.1887902 , 4.71238898, 5.23598776, 5.75958653])]),
                             ('y', [SUNSHINE_HOURS])]),
        "ticklabels": OrderedDict([('x', MONTHS),
                                   ('y', ['0', '78', '156', '234', '312'])]),
        "axislabels": OrderedDict([('x', ''), ('y', '')]),
        "axistypes": OrderedDict([('x', 'datetime'), ('y', 'linear')]),
    },
    scatter_gaussian: {
        "type": "scatter",
        "data": OrderedDict([('x', [BLUE_RANDOM_GAUSSIAN_2D[:, 0], ORANGE_RANDOM_GAUSSIAN_2D[:, 0]]),
                             ('y', [BLUE_RANDOM_GAUSSIAN_2D[:, 1], ORANGE_RANDOM_GAUSSIAN_2D[:, 1]])]),
        "ticklabels": None, #OrderedDict([('x', ['−10', '−8', '−6', '−4', '−2', '0', '2', '4', '6', '8', '10']),
                      #             ('y', ['−6', '−4', '−2', '0', '2', '4', '6', '8', '10'])]),
        "axislabels": OrderedDict([('x', 'Random Gaussian x'), ('y', 'Random Gaussian y')]),
        "axistypes": OrderedDict([('x', 'linear'), ('y', 'linear')]),
    },
    strip_gaussian: {
        "type": "scatter",
        "data": OrderedDict([('x', [BLUE_RANDOM_GAUSSIAN_2D[:, 0], ORANGE_RANDOM_GAUSSIAN_2D[:, 0]]),
                             ('y', [np.repeat(0, len(BLUE_RANDOM_GAUSSIAN_2D)), np.repeat(1, len(ORANGE_RANDOM_GAUSSIAN_2D))])]),
        "ticklabels": None, #OrderedDict([('x', ['−10', '−8', '−6', '−4', '−2', '0', '2', '4', '6', '8', '10']),
                     #('y', ['The blue dots', 'The orange dots'])]),
        "axislabels": OrderedDict([('x', 'Random Gaussian x'), ('y', '')]),
        "axistypes": OrderedDict([('x', 'linear'), ('y', 'categorical')]),
    },
    heatmap_gaussian: {
        "type": "heatmap",
        "data": None,
        "ticklabels": None,
        "axislabels": OrderedDict([('x', ''), ('y', ''), ('z', 'Number of gaussian points')]),
        "axistypes": OrderedDict([('x', 'linear'), ('y', 'linear'), ('z', 'linear')]),
    },
    contour_gaussian: {
        "type": "contour",
        "data": OrderedDict([('x', []), ('y', [])]),
        "ticklabels": None, #OrderedDict([('x', ['−10', '−8', '−6', '−4', '−2', '0', '2', '4', '6']),
                      #             ('y', ['−4', '−2', '0', '2', '4', '6', '8', '10'])]),
        "axislabels": OrderedDict([('x', ''), ('y', '')]),
        "axistypes": OrderedDict([('x', 'linear'), ('y', 'linear')]),
    },
    boxplot_anscombs: {
        "type": "boxplot",
        "data": OrderedDict(),
        "ticklabels": None, #OrderedDict([('x', ['0', '1', '2', '3']), ('y', ['2', '4', '6', '8', '10', '12', '14'])]),
        "axislabels": OrderedDict([('x', 'Subplot Num'), ('y', "y Value in Anscombe's Quartet")]),
        "axistypes": OrderedDict([('x', 'linear'), ('y', 'linear')]),
    },
}