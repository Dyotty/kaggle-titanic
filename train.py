# -*- coding: utf-8 -*-

import csv
import numpy as np
from sklearn.preprocessing import StandardScaler


def read_csv(file_path):
    with open(filepath) as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)