import os
import sys
import types
import importlib
import importlib.resources
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import numpy as np
import seaborn as sns
from joblib import delayed, Parallel
import altair as alt
from statistics import harmonic_mean, geometric_mean

PLOTS_DIR = Path(__file__).resolve().parent / "new_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

N_JOBS = min(8, os.cpu_count() or 1)

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
)
os.environ.setdefault(
    "PYTHONWARNINGS",
    r"ignore:pkg_resources is deprecated as an API\..*:UserWarning",
)

try:
    import pkg_resources
except ModuleNotFoundError:
    pkg_resources = types.ModuleType("pkg_resources")

    def resource_stream(package_or_requirement, resource_name):
        if isinstance(package_or_requirement, str):
            package = importlib.import_module(package_or_requirement)
        else:
            package = package_or_requirement
        return importlib.resources.files(package).joinpath(resource_name).open("rb")

    pkg_resources.resource_stream = resource_stream
    sys.modules["pkg_resources"] = pkg_resources

from pybdm import BDM
from pybdm import PartitionRecursive

# Load the CSV and apply name mapping
# name_mapping = create_model_name_mapping()
# Repo-relative so the pipeline runs from a clean checkout on any machine.
# The former sibling file seriesWithLLMs_ext_Aug2025.csv was byte-identical to
# this one (same MD5) and has been removed.
DATA_DIR = Path(__file__).resolve().parent
exp_path = DATA_DIR / "seriesWithLLMs_ext_Dic2025.csv"
bin_seq_df_input = pd.read_csv(exp_path, encoding='latin-1')
# bin_seq_df_input = rename_model_columns(bin_seq_df_input_raw, name_mapping)

bin_seqs = bin_seq_df_input.iloc[:100,:]
int_seqs_1 = bin_seq_df_input.iloc[100:130,:]
int_seqs_2 = bin_seq_df_input.iloc[130:160,:]
int_seqs_3 = bin_seq_df_input.iloc[160:,:]

bin_seq_df=bin_seqs

models = [
    "chatgpt_4.5",
    "o1_mini",
    "claude_3.7",
    "claude_3.5",
    "o1_preview",
    "gemini",
    "cursor_small",
    "gpt_4o_mini",
    "mistral",
    "qwen",
    "deepseek",
    "llama_4_scout",
    "grok_3",
    "qwen3",
    "chatgpt_5",
    "grok4",
    "deepseek_r1_0528",
    "opus_4",
    "mistral_large2405",
    "gemini_2.5_pro",
    "claude_sonnet_4",
    "meta",
    "gpt_4o",
    "grok_4.1",
    "gpt-5.2",
    "claude-4.5",
    "gemini-3-pro",
    "mistral-large-3",
]

def ascii_to_binary_list(text):
    binary_list = []
    for char in text:
        # Convert each character to its binary representation
        binary_representation = format(ord(char), '08b')
        # Extend the binary_list with the numerical digits of the binary representation
        binary_list.extend([int(bit) for bit in binary_representation])
    return np.array(binary_list).astype(np.int8)

def tst_calc(v1,v2):
    v3 = (v2)*np.array([1,0.1,0.01])
    return np.sum(v1[:3]*v3)




def process_sequence(seq_str):
    """
    Process a sequence string to prepare it for BDM calculation.
    - If it's already a binary sequence (0s and 1s), split by commas and convert to int8
    - If it's a numeric sequence, convert each number to binary digits and flatten
    """
    elements = seq_str.split(",")
    # Check if all elements are binary (0 or 1)
    is_binary = all(e.strip() in ['0', '1'] for e in elements)
    
    if is_binary:
        # Already binary, just convert to int8 array
        return np.array(elements).astype(np.int8)
    else:
        # Convert numeric sequence to binary representation
        binary_digits = []
        for num in elements:
            try:
                # Convert to integer and then to binary string (removing '0b' prefix)
                bin_str = bin(int(num.strip()))[2:]
                # Convert each binary digit to int and add to list
                binary_digits.extend([int(digit) for digit in bin_str])
            except ValueError:
                # If conversion fails, use empty list
                pass
        return np.array(binary_digits).astype(np.int8)
    
def get_model_display_name(model_key):
    """Map internal model key to display name using the predefined mapping."""
    model_name_mapping = {
        'gpt_4o': 'ChatGPT-4o',
        'claude_3.5': 'Claude-3.5',
        'gpt_4o_mini': 'ChatGPT-4o-Mini',
        'cursor_small': 'Cursor-Small',
        'gemini': 'Gemini',
        'meta': 'Meta',
        'o1_mini': 'o1-Mini',
        'o1_preview': 'o1-Preview',
        'mistral': 'Mistral',
        'qwen': 'Qwen',
        'grok_3': 'Grok-3',
        'deepseek': 'DeepSeek',
        'claude_3.7': 'Claude-3.7',
        'chatgpt_4.5': 'ChatGPT-4.5',
        'opus_4': 'Claude-Opus-4',
        'claude_sonnet_4': 'Claude-Sonnet-4',
        'gemini_2.5_pro': 'Gemini-2.5-Pro',
        'mistral_large2405': 'Mistral-Large-2405',
        'qwen3': 'Qwen-3',
        'deepseek_r1_0528': 'DeepSeek-R1-0528',
        'grok4': 'Grok-4',
        'llama_4_scout': 'Llama-4-Scout',
        'chatgpt_5': 'ChatGPT-5',
        "grok_4.1": "Grok-4.1",
        "gpt-5.2": "ChatGPT-5.2",
        "claude-4.5": "Claude-4.5",
        "gemini-3-pro": "Gemini-3-Pro",
        "mistral-large-3": "Mistral-Large-3",  
    }
    return model_name_mapping.get(model_key, model_key)

def save_altair_chart(chart, file_base):
    if importlib.util.find_spec("vl_convert") is None:
        raise ValueError(
            "Saving Altair charts as PDF/PNG requires vl-convert-python. Install with: python -m pip install vl-convert-python"
        )
    alt.data_transformers.enable("default")
    chart.save(str(PLOTS_DIR / f"{file_base}.pdf"))
    chart.save(str(PLOTS_DIR / f"{file_base}.png"), scale_factor=3)

test_vals=[]
bdm = BDM(ndim=1,partition=PartitionRecursive)
for mdl in models:#[:-1]:
    display_name = get_model_display_name(mdl)
    print('processing model: ', display_name)
    sep_df= bin_seq_df[['sequence',f'{mdl}-formula',f'{mdl}-formula-correctness',f'{mdl}-formula-ordinal',f'{mdl}-formula-copy_seq']].copy()
    sep_df["bdm_formula"]=[bdm.nbdm(ascii_to_binary_list(x)) for x in sep_df[f'{mdl}-formula'].to_numpy()]
    sep_df["bdm_input"]=[bdm.nbdm(ascii_to_binary_list(x)) for x in sep_df['sequence'].to_numpy()]
    
    df_c_n_n = sep_df[sep_df[f"{mdl}-formula-correctness"] & ~sep_df[f"{mdl}-formula-ordinal"] & ~sep_df[f"{mdl}-formula-copy_seq"]]
    df_c_o = sep_df[sep_df[f"{mdl}-formula-correctness"] & sep_df[f"{mdl}-formula-ordinal"]]
    df_c_p = sep_df[sep_df[f"{mdl}-formula-correctness"] & sep_df[f"{mdl}-formula-copy_seq"]]
    df_i = sep_df[~sep_df[f"{mdl}-formula-correctness"]]
    tot_elements = np.sum([len(x) for x in [df_c_n_n,df_c_o,df_c_p,df_i]])
    v1=np.array([len(x)/tot_elements for x in [df_c_n_n,df_c_o,df_c_p,df_i]])
    v2 = []
    for x in [df_c_n_n,df_c_o,df_c_p]:
        datax = np.tanh(((x["bdm_input"])/(x["bdm_formula"])).to_numpy())
        if len(datax)>0:
            v2.append(harmonic_mean(datax))
        else:
            v2.append(0)
    v2 = (np.nan_to_num(np.array(v2)))
    tst = tst_calc(v1,v2)
    test_vals.append([display_name]+list(v1)+list(v2)+[tst])



df_ranking = pd.DataFrame(test_vals)
df_ranking.columns = ["Model","p1","p2","p3","p4","r1","r2","r3","tst"]
np.round(df_ranking.sort_values(by=['tst'],ascending=False).set_index(["Model"]),3)


df_ranking = pd.DataFrame(test_vals)
df_ranking.columns = ["Model","p1","p2","p3","p4","r1","r2","r3","tst"]
np.round(df_ranking.sort_values(by=['tst'],ascending=False).set_index(["Model"]),3)

new_row = {'Model': 'ASI', 'p1': 1, 'p2': 0, 'p3': 0, 'p4': 0, 'r1': 1, 'r2': 0, 'r3': 1, 'tst': 1}
df_ranking2 = pd.concat([df_ranking, pd.DataFrame([new_row])], ignore_index=True)
df_ranking2 = df_ranking2.sort_values(by='tst', ascending=False)
np.round(df_ranking2,3)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
# Sort values in descending order
sorted_df = df_ranking2.sort_values('tst', ascending=False)
# Create bar plot using Model column with default color cycle
bars = plt.bar(range(len(sorted_df)), sorted_df['tst'])
# Set different colors for each bar using default color cycle
for i, bar in enumerate(bars):
    bar.set_color(f'C{i}')
plt.grid(True)
plt.title('Ranking by SuperARC-seq')
plt.xlabel('Model')
plt.ylabel('SuperARC-seq')
# Set y-axis to log scale
plt.yscale('log')
# Set x-tick labels with model names
plt.xticks(range(len(sorted_df)), sorted_df['Model'], rotation=45, ha='right')
plt.tight_layout()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
# Sort values in descending order
sorted_df = df_ranking2.sort_values('tst', ascending=False)
# Create bar plot using Model column with default color cycle
bars = plt.bar(range(len(sorted_df)), sorted_df['tst'])
# Set different colors for each bar using default color cycle
for i, bar in enumerate(bars):
    bar.set_color(f'C{i}')
plt.grid(True)
plt.title('Ranking by SuperARC-seq')
plt.xlabel('Model')
plt.ylabel('SuperARC-seq')

# Set x-tick labels with model names
plt.xticks(range(len(sorted_df)), sorted_df['Model'], rotation=45, ha='right')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "figure05.pdf", bbox_inches="tight")
plt.savefig(PLOTS_DIR / "figure05.png", bbox_inches="tight", dpi=600)



dfs_study = [bin_seqs, int_seqs_1, int_seqs_2, int_seqs_3]
labels_study = ["Binary", "Integers Type 1","Integers Type 2","Integers Type 3"]

test_vals=[]
bdm = BDM(ndim=1,partition=PartitionRecursive)
for ll,dd in zip(labels_study,dfs_study):  
    for mdl in models:
        sep_df= dd[['sequence',f'{mdl}-formula',f'{mdl}-formula-correctness',f'{mdl}-formula-ordinal',f'{mdl}-formula-copy_seq']].copy()

        df_c_n_n = sep_df[sep_df[f"{mdl}-formula-correctness"] & ~sep_df[f"{mdl}-formula-ordinal"] & ~sep_df[f"{mdl}-formula-copy_seq"]]
        df_c_o = sep_df[sep_df[f"{mdl}-formula-correctness"] & sep_df[f"{mdl}-formula-ordinal"]]
        df_c_p = sep_df[sep_df[f"{mdl}-formula-correctness"] & sep_df[f"{mdl}-formula-copy_seq"]]
        df_i = sep_df[~sep_df[f"{mdl}-formula-correctness"]]
        tot_elements = np.sum([len(x) for x in [df_c_n_n,df_c_o,df_c_p,df_i]])
        v1=np.array([len(x)/tot_elements for x in [df_c_n_n,df_c_o,df_c_p,df_i]])
        test_vals.append([mdl]+list(v1)+[ll])

df_stack = pd.DataFrame(test_vals,columns = ["Model","p1","p2","p3","p4","Type"]);

dfs_altair=[]
for pp in ["p1","p2","p3","p4"]:
    df_filt_stack = df_stack.loc[:,pp].to_numpy()
    num_models = len(models)
    df_itermv = pd.DataFrame(df_filt_stack.reshape(len(df_filt_stack)//num_models, num_models, order='F')).T   
    df_itermv.index = models
    df_itermv.columns = labels_study
    dfs_altair.append(df_itermv)

def prep_df(df, name):
    df = df.stack().reset_index()
    df.columns = ['c1', 'c2', 'values']
    df['Prob'] = name
    return df

lst_final_df = []
for pp,nm in enumerate(["p₁","p₂","p₃","p₄"]):
    lst_final_df.append(prep_df(dfs_altair[pp], nm))

df_final = pd.concat(lst_final_df,ignore_index=True)

# Map model keys to display names
df_final['c1'] = df_final['c1'].apply(get_model_display_name)

# ==========================================
# FIGURE 06 (Proposal 3): Faceted Dot Plot
# ==========================================
ALT_AXIS_LABEL_FONT = 16
ALT_AXIS_TITLE_FONT = 18
ALT_LEGEND_LABEL_FONT = 16
ALT_LEGEND_TITLE_FONT = 18

prob_colors = alt.Scale(domain=["p₁", "p₂", "p₃", "p₄"], range=['#96ceb4', '#ffcc5c', '#ff6f69', '#da680f'])

plot06 = alt.Chart(df_final).mark_circle(size=80, opacity=0.8).encode(
    y=alt.Y('c1:N', title=None, sort=alt.EncodingSortField(field='values', op='sum', order='descending')),
    x=alt.X('values:Q', title="Probability", scale=alt.Scale(domain=[0, 1])),
    color=alt.Color('Prob:N', scale=prob_colors, title="Probability Type"),
    column=alt.Column('c2:N', title="", header=alt.Header(labelFontSize=ALT_AXIS_LABEL_FONT, titleFontSize=ALT_AXIS_TITLE_FONT))
).properties(
    width=220, height=620
).configure_axis(
    labelFontSize=ALT_AXIS_LABEL_FONT, titleFontSize=ALT_AXIS_TITLE_FONT, grid=True
).configure_legend(
    labelFontSize=ALT_LEGEND_LABEL_FONT, titleFontSize=ALT_LEGEND_TITLE_FONT, orient='top'
)

save_altair_chart(plot06, "figure06")

test_vals_2=[]
bdm = BDM(ndim=1,partition=PartitionRecursive)
for ll,dd in zip(labels_study,dfs_study):  
    for mdl in models:
        sep_df= dd[['sequence',f'{mdl}-formula',f'{mdl}-formula-correctness',f'{mdl}-formula-ordinal',f'{mdl}-formula-copy_seq']].copy()

        sep_df["bdm_formula"]=[bdm.nbdm(ascii_to_binary_list(x)) for x in sep_df[f'{mdl}-formula'].to_numpy()]
        sep_df["bdm_input"]=[bdm.nbdm(ascii_to_binary_list(x)) for x in sep_df['sequence'].to_numpy()]

        df_c_n_n = sep_df[sep_df[f"{mdl}-formula-correctness"] & ~sep_df[f"{mdl}-formula-ordinal"] & ~sep_df[f"{mdl}-formula-copy_seq"]]
        df_c_o = sep_df[sep_df[f"{mdl}-formula-correctness"] & sep_df[f"{mdl}-formula-ordinal"]]
        df_c_p = sep_df[sep_df[f"{mdl}-formula-correctness"] & sep_df[f"{mdl}-formula-copy_seq"]]
        df_i = sep_df[~sep_df[f"{mdl}-formula-correctness"]]
        tot_elements = np.sum([len(x) for x in [df_c_n_n,df_c_o,df_c_p,df_i]])
        v1=np.array([len(x)/tot_elements for x in [df_c_n_n,df_c_o,df_c_p,df_i]])
        v2 = []
        for x in [df_c_n_n,df_c_o,df_c_p]:
            datax = np.tanh(((x["bdm_input"])/(x["bdm_formula"])).to_numpy())
            if len(datax)>0:
                v2.append(harmonic_mean(datax))
            else:
                v2.append(0)
        v2 = (np.nan_to_num(np.array(v2)))
        tst = tst_calc(v1,v2)
        test_vals_2.append([mdl]+list(v2)+[tst,ll])

df_stack_2 = pd.DataFrame(test_vals_2,columns = ["Model","r1","r2","r3","tst","Type"]);

dfs_altair_2=[]
for pp in ["tst"]:
    df_filt_stack = df_stack_2.loc[:,pp].to_numpy()
    num_models = len(models)
    df_itermv = pd.DataFrame(df_filt_stack.reshape(len(df_filt_stack)//num_models, num_models)).T
    df_itermv.index = models
    df_itermv.columns = labels_study
    dfs_altair_2.append(df_itermv)

def prep_df(df, name):
    df = df.stack().reset_index()
    df.columns = ['c1', 'c2', 'values']
    df['Metric'] = name
    return df

lst_final_df_2 = []
for pp,nm in enumerate(["𝜑"]):
    lst_final_df_2.append(prep_df(dfs_altair_2[pp], nm))

df_final_2 = pd.concat(lst_final_df_2,ignore_index=True)

df_final_2['c1'] = df_final_2['c1'].apply(get_model_display_name)

# ==========================================
# FIGURE 07 (Proposal 3): Unified Dot Plot
# ==========================================
seq_colors = alt.Scale(domain=labels_study, range=['#96ceb4', '#ffcc5c', '#ff6f69', '#da680f'])

# Get the model order from Figure 06 (sorted by sum of values descending)
model_order_fig06 = df_final.groupby('c1')['values'].sum().sort_values(ascending=False).index.tolist()

plot07 = alt.Chart(df_final_2).mark_circle(size=120, opacity=0.85).encode(
    y=alt.Y('c1:N', title=None, sort=model_order_fig06),
    x=alt.X('values:Q', title="𝜑 (Harmonic Mean Ratio)"),
    color=alt.Color('c2:N', title="Sequence Type", scale=seq_colors),
    tooltip=['c1', 'c2', 'values']
).properties(
    width=700, height=620
).configure_axis(
    labelFontSize=ALT_AXIS_LABEL_FONT, titleFontSize=ALT_AXIS_TITLE_FONT, grid=True
).configure_legend(
    labelFontSize=ALT_LEGEND_LABEL_FONT, titleFontSize=ALT_LEGEND_TITLE_FONT, orient='top'
)

save_altair_chart(plot07, "figure07")

# Your existing inner_loop_function and data processing code here
# (keeping it the same as your original)

def inner_loop_function(models,bin_seq_df):
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
    )
    bdm = BDM(ndim=1,partition=PartitionRecursive)
    dict_tst={}
    for mdl in models:#[:-1]:
        sep_df= bin_seq_df[['sequence',f'{mdl}-formula',f'{mdl}-formula-correctness',f'{mdl}-formula-ordinal',f'{mdl}-formula-copy_seq']].copy()
        sep_df["bdm_formula"]=[bdm.nbdm(ascii_to_binary_list(x)) for x in sep_df[f'{mdl}-formula'].to_numpy()]
        sep_df["bdm_input"]=[bdm.nbdm(ascii_to_binary_list(x)) for x in sep_df['sequence'].to_numpy()]
        df_c_n_n = sep_df[sep_df[f"{mdl}-formula-correctness"] & ~sep_df[f"{mdl}-formula-ordinal"] & ~sep_df[f"{mdl}-formula-copy_seq"]]
        df_c_o = sep_df[sep_df[f"{mdl}-formula-correctness"] & sep_df[f"{mdl}-formula-ordinal"]]
        df_c_p = sep_df[sep_df[f"{mdl}-formula-correctness"] & sep_df[f"{mdl}-formula-copy_seq"]]
        df_i = sep_df[~sep_df[f"{mdl}-formula-correctness"]]
        tot_elements = np.sum([len(x) for x in [df_c_n_n,df_c_o,df_c_p,df_i]])
        v1=np.array([len(x)/tot_elements for x in [df_c_n_n,df_c_o,df_c_p,df_i]])
        v2 = []
        for x in [df_c_n_n,df_c_o,df_c_p]:
            datax = np.tanh(((x["bdm_input"])/(x["bdm_formula"])).to_numpy())
            if len(datax)>0:
                v2.append(harmonic_mean(datax))
            else:
                v2.append(0)
        v2 = (np.nan_to_num(np.array(v2)))
        tst = tst_calc(v1,v2)
        dict_tst[mdl] = [tst]
    return dict_tst

# np.random.seed() seeds the LEGACY global RNG; np.random.default_rng() with no
# argument builds a fresh Generator from OS entropy and ignores it entirely, so
# the bootstrap was never reproducible. Seeding the Generator itself fixes that.
# Consequence: o1-Mini's displayed mean becomes 0.034 rather than the published
# 0.035. Its exact (non-bootstrap) test score is 0.034380, so 0.034 is the
# correctly rounded value and the published figure showed the noisier draw.
rng = np.random.default_rng(42)
tst_bootstrap=[]
for sz in [25,50,75,100]:
    inpts_bts = []
    for rep_id in range(100):
        idx_v = rng.integers(low=0, high=100, size=sz,dtype = int)
        inpts_bts.append([models,bin_seqs.iloc[idx_v]])
    calcs_lst = Parallel(n_jobs=N_JOBS)(delayed(inner_loop_function)(*x) for x in inpts_bts)
    dict_tst={}
    for mdl in models:
        dict_tst[mdl]=[]
    for dict_indiv in calcs_lst:
        for ddd in dict_indiv.keys():
            dict_tst[ddd]=dict_tst[ddd]+dict_indiv[ddd]
    tst_bootstrap.append(dict_tst)

lst_dfs=[]
for pp,sz in enumerate([25,50,75,100]): 
    df_interm = pd.DataFrame(tst_bootstrap[pp]).T.reset_index()
    df_interm = df_interm.melt(id_vars=['index'])
    df_interm.loc[:,'variable']=sz
    lst_dfs.append(df_interm)

df_bts = pd.concat(lst_dfs)
df_bts.columns = ["Model","# of Sequences","Test Score"]
# Map model names to display names
df_bts['Model'] = df_bts['Model'].apply(get_model_display_name)

# Calculate mean test score for each model
model_means = df_bts.groupby('Model')['Test Score'].mean().sort_values(ascending=False)

# Assign broad performance category
def assign_broad_tier(model_name):
    mean_score = model_means[model_name]
    if mean_score >= 0.030:
        return 'High'
    elif mean_score >= 0.007:
        return 'Medium'
    else:
        return 'Low'

df_bts['Broad_Tier'] = df_bts['Model'].apply(assign_broad_tier)

# Create the plot with 3 main tiers, but use offset plotting within each tier
FIG08_TITLE_FONT = 20
FIG08_AXIS_LABEL_FONT = 17
FIG08_TICK_FONT = 15
FIG08_LEGEND_FONT = 15
FIG08_ANNOT_FONT = 14

fig, axes = plt.subplots(3, 1, figsize=(18, 16), sharex=True)

tiers = ['High', 'Medium', 'Low']
tier_titles = {
    'High': 'High Performance (≥0.030)',
    'Medium': 'Medium Performance (0.007-0.030)',
    'Low': 'Low Performance (<0.007)'
}

# Create color palette
all_models = sorted(df_bts['Model'].unique())
colors = sns.color_palette("husl", len(all_models))
model_color_map = dict(zip(all_models, colors))

# Different markers
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'P', 'h', '<', '>', '8']

for idx, tier in enumerate(tiers):
    tier_data = df_bts[df_bts['Broad_Tier'] == tier]
    tier_models = sorted(tier_data['Model'].unique(), 
                        key=lambda x: model_means[x], 
                        reverse=True)
    
    n_models = len(tier_models)
    
    if n_models == 0:
        axes[idx].text(0.5, 0.5, 'No models in this tier', 
                      ha='center', va='center', transform=axes[idx].transAxes)
        continue
    
    # Calculate vertical offset to separate overlapping lines
    # Models will be slightly offset vertically for visibility
    for model_id, model in enumerate(tier_models):
        model_data = tier_data[tier_data['Model'] == model]
        
        # Calculate statistics
        mean_scores = model_data.groupby('# of Sequences')['Test Score'].mean()
        std_scores = model_data.groupby('# of Sequences')['Test Score'].std()
        ci_lower = model_data.groupby('# of Sequences')['Test Score'].quantile(0.25)
        ci_upper = model_data.groupby('# of Sequences')['Test Score'].quantile(0.75)
        
        # Apply small vertical offset based on position in sorted list
        # This spreads overlapping lines slightly
        offset_factor = 0.0001 if tier == 'High' else (0.00005 if tier == 'Medium' else 0.000002)
        vertical_offset = (model_id - n_models/2) * offset_factor
        
        # CHANGE 4: Create label with average in parentheses
        model_avg = model_means[model]
        label_with_avg = f"{model} ({model_avg:.3f})"
        
        # CHANGE 2: Increased markersize from 7 to 9.1 (30% bigger: 7 * 1.3 = 9.1)
        axes[idx].plot(
            mean_scores.index,
            mean_scores.values + vertical_offset,
            marker=markers[model_id % len(markers)],
            markersize=9.1,
            linewidth=2,
            label=label_with_avg,
            color=model_color_map[model],
            markeredgewidth=0.7,
            markeredgecolor='white',
            alpha=0.85
        )
        
        # Add very subtle confidence band (without offset for accuracy)
        axes[idx].fill_between(
            mean_scores.index,
            ci_lower.values,
            ci_upper.values,
            alpha=0.08,
            color=model_color_map[model]
        )
    
    # Formatting
    axes[idx].set_title(tier_titles[tier], fontsize=FIG08_TITLE_FONT, fontweight='bold', pad=12)
    axes[idx].set_ylabel('Test Score', fontsize=FIG08_AXIS_LABEL_FONT)
    axes[idx].tick_params(axis='both', which='major', labelsize=FIG08_TICK_FONT)
    axes[idx].grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    
    # CHANGE 1: Always place High tier legend outside, adjust other tiers accordingly
    # CHANGE 3: Increased legend fontsize by 25% (9 * 1.25 = 11.25, 8 * 1.25 = 10)
    if tier == 'High':
        # Always place High performance legend outside the plot area
        axes[idx].legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                        fontsize=FIG08_LEGEND_FONT, framealpha=0.95, ncol=1)
    elif n_models <= 6:
        axes[idx].legend(loc='best', fontsize=FIG08_LEGEND_FONT, framealpha=0.95, ncol=1)
    else:
        axes[idx].legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                        fontsize=FIG08_LEGEND_FONT, framealpha=0.95, ncol=1)
    
    # Add model count annotation
    axes[idx].text(0.02, 0.02, f'{n_models} model{"s" if n_models != 1 else ""}',
                  transform=axes[idx].transAxes,
                  fontsize=FIG08_ANNOT_FONT, verticalalignment='bottom',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

axes[-1].set_xlabel('# of Sequences', fontsize=FIG08_AXIS_LABEL_FONT)
axes[-1].set_xticks([25, 50, 75, 100])

plt.tight_layout(rect=[0, 0, 0.72, 1])
plt.savefig(PLOTS_DIR / "figure08.pdf", bbox_inches="tight")
plt.savefig(PLOTS_DIR / "figure08.png", bbox_inches="tight", dpi=600)

# Print summary statistics
print("\nModel Performance Summary by Tier:")
print("=" * 80)
for tier in tiers:
    tier_models = df_bts[df_bts['Broad_Tier'] == tier]['Model'].unique()
    if len(tier_models) > 0:
        print(f"\n{tier_titles[tier]} ({len(tier_models)} models):")
        for model in sorted(tier_models, key=lambda x: model_means[x], reverse=True):
            mean = model_means[model]
            std = df_bts[df_bts['Model'] == model]['Test Score'].std()
            print(f"  • {model:30s}  Mean: {mean:.6f}  Std: {std:.6f}")
