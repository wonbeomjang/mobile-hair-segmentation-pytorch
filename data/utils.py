import os


def check_data(data_folder):
    masks = set(os.listdir(f'{data_folder}/masks/'))
    image = set(os.listdir(f'{data_folder}/images/'))

    intersection = masks.intersection(image)
    union = masks.union(image)
    print(f"[!] {len(union) - len(intersection)} of {len(union)} images doesn't have mask")

    intersection = list(intersection)

    return intersection