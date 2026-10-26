# bpy-free, the dev cli loads this file by path

# the upper bound on distortion per priority
PRIORITY_VALUES = {"LESS_STRETCH": "4.05", "BALANCED": "4.2", "FEWER_SEAMS": "5.0"}
# the seam cost on painted faces per weight level
SEAM_WEIGHTS = {1: "25", 2: "50", 3: "100", 4: "150", 5: "200"}


def bound_and_weight_args(priority, weight_level):
    return ["-u", PRIORITY_VALUES[priority], "-s", SEAM_WEIGHTS[weight_level]]
