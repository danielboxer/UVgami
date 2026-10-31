PRIORITY_VALUES = {"LESS_STRETCH": "4.05", "BALANCED": "4.2", "FEWER_SEAMS": "5.0"}
SEAM_WEIGHTS = {1: "25", 2: "50", 3: "100", 4: "150", 5: "200"}
SPLIT_TARGETS = {"LESS_STRETCH": 0.6, "BALANCED": 0.5, "FEWER_SEAMS": 0.4}


def bound_and_weight_args(priority, weight_level):
    return ["-u", PRIORITY_VALUES[priority], "-s", SEAM_WEIGHTS[weight_level]]
