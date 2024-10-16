import numpy as np
from scipy.stats import linregress

sim_metric_arrs = np.load("./caption_sim_metric_to_arr.npy", allow_pickle=True).item()
blip_metric_arrs = np.load("./caption_blip_metric_to_arr.npy", allow_pickle=True).item()

linear_fits = {
    "human_sim": {},
    "blipscores": {}
}

print("Human similarity len lines of fit ---------------------------------------")
sim_len_arrs = sim_metric_arrs["lens"]
for metric_name, cap_scores in sim_metric_arrs.items():
    if metric_name != "lens":
        linfit = linregress(sim_len_arrs, cap_scores)
        linear_fits["human_sim"][metric_name] = linfit
        print(f"{metric_name}: {linfit}")

print("BLIP len lines of fit ---------------------------------------")
blip_len_arrs = blip_metric_arrs["lens"]
for metric_name, cap_scores in blip_metric_arrs.items():
    if metric_name != "lens":
        linfit = linregress(blip_len_arrs, cap_scores)
        linear_fits["blipscores"][metric_name] = linfit
        print(f"{metric_name}: {linfit}")

np.save("./len_linear_fits", linear_fits)