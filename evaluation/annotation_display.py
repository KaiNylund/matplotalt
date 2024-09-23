import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os

vistext_combined_captions_df = pd.read_json("./vistext_eval/vistext_id_to_combined_captions.jsonl", orient='records', lines=True)
gallery_combined_captions_df = pd.read_json("./matplotlib_gallery/mpl_gallery_combined_captions_shuffled.jsonl", orient='records', lines=True)

def display_vistext_img_and_captions(row):
    os.system('cls')
    vistext_img = matplotlib.image.imread(f"./vistext_eval/matplotlib_ver_imgs/{row['image_id']}.png")
    print("------------------------------------------------------------------------------")
    print(f"Image ID: {row['image_id']}")
    print("------------------------------------------------------------------------------")
    for i, hcap in enumerate(row["human"]):
        print(f"Human caption {i}: {hcap}")
        print("------------------------------------------------------------------------------")
    print(f"Heuristic: {row['heuristic'][0].replace('This description was generated by a language model. ', '')}")
    print("------------------------------------------------------------------------------")
    print(f"GPT4 L3: {row['gpt-4-turbo-L3'][0].replace('This description was generated by a language model. ', '')}")
    print("------------------------------------------------------------------------------")
    print(f"GPT4 heuristic L3: {row['gpt-4-turbo-alt-L3'][0].replace('This description was generated by a language model. ', '')}")
    print("------------------------------------------------------------------------------")
    print(f"GPT4 table L3: {row['gpt-4-turbo-table-L3'][0].replace('This description was generated by a language model. ', '')}")
    print("------------------------------------------------------------------------------")
    print(f"GPT4 table + heuristic L3: {row['gpt-4-turbo-table-alt-L3'][0].replace('This description was generated by a language model. ', '')}")
    print("------------------------------------------------------------------------------")
    plt.imshow(vistext_img)
    plt.show()

def display_gallery_img_and_captions(row):
    os.system('cls')
    gallery_img = matplotlib.image.imread(f"./matplotlib_gallery/alt_figs/nb_{row['figure_id']}.jpg")
    print("------------------------------------------------------------------------------")
    print(f"Image ID: {row['figure_id']}")
    print("------------------------------------------------------------------------------")
    print(f"Heuristic: {row['heuristic']}")
    print("------------------------------------------------------------------------------")
    print(f"GPT4 L4: {row['gpt-4-turbo-L4-300']}")
    print("------------------------------------------------------------------------------")
    print(f"GPT4 heuristic L4: {row['gpt-4-turbo-alt-L4-300']}")
    print("------------------------------------------------------------------------------")
    plt.imshow(gallery_img)
    plt.show()

#print(vistext_combined_captions_df)

#gallery_combined_captions_df.apply(display_gallery_img_and_captions, axis=1)
vistext_combined_captions_df.apply(display_vistext_img_and_captions, axis=1)