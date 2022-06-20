import numpy as np
import pandas as pd
import time
import random
import csv
import ast

from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis
from PCAfold.styles import *

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import plotly as plty
from plotly import express as px
from plotly import graph_objects as go

from scipy.stats import pearsonr
from scipy.stats import spearmanr

# Common settings: - - - - - - - - - - - - - - - - - - - - 
random_seed = 100
norm = 'cumulative'
penalty_function = 'log-sigma-over-peak'
power = 1
vertical_shift = 1
bandwidth_values = np.logspace(-7, 3, 200)