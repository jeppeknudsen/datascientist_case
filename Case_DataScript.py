# %% Import af pakker

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

# Sæt plot font
plt.rcParams.update({'font.size': 12})

# %% Data import
# NB: Den givne fil er konverteret til CSV-format i Excel, så søjler registreres individuelt i pandas

# Originalt filnavn - bruges ikke her!
original_file_name = 'DataScientist-Case-Dataset.xlsx'

# Fil manuelt konverteret til csv til nemmere håndtering i Pandas
csv_file_name = 'DataScientist-Case-Dataset.csv' 


# Sætter directory til path for dette script
data_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(data_dir)

# Import af data i dataframe
df = pd.DataFrame(pd.read_csv(csv_file_name, index_col='customer_id'))


# Print info for at tjekke om alt ser ok ud
print(df.info())

# %% Kør describe for at få overblik
df.describe()


# %% Del op i kategorisk og numerisk data

# Liste over kategoriske variable
cat_list = ['converted', 'customer_segment', 'gender', 'credit_account_id', 'branch']

# Liste over numeriske variable
num_list = ['age', 'related_customers', 'family_size', 'initial_fee_level']

# Individuelle dataframes til kategoriske og numeriske data
df_cat = df[cat_list]
df_num = df[num_list]


# %% NUMERIC
# Initial plots for at få et overblik over numerisk data


for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i, fontsize=18)
    plt.xlabel(i, fontsize=18)
    plt.ylabel('Frekvens', fontsize=18)
    plt.show()

# %% Kig på converted på tværs af numeriske parametres gennemsnit

pd.pivot_table(df, index='converted', values=num_list)

# %% CATEGORICAL
# Indledende plotning af kategorisk data

for j in df_cat.columns:
    if j != 'credit_account_id': # Undgå unyttigt og krævende plot af account_id
        sns.barplot(x=df_cat[j].value_counts().index, y=df_cat[j].value_counts()).set_title(j)
        plt.show()



# %% Isoleret plot af konverterede
ax0 = sns.barplot(x=df_cat['converted'].value_counts().index, y=df_cat['converted'].value_counts())
ax0.set_xticklabels(['Ikke konverteret', 'Konverteret'], fontsize=14)
ax0.set_ylabel('Count', fontsize=14)
ax0.set_title('Konverterede vs ikke-konverterede', fontsize=14)



# %% Bestemmelse af konvertionsrate

n_total = df.shape[0]
n_convert = len(df[df['converted']==1])
convert_rate = n_convert/len(df)

print(f'''
Antal konverterede er{n_convert} ud af {n_total} brugere
Konvertionsraten er {convert_rate:.3f}
''')



# %% Registrering af Nan-values

print('Number of Nan-values in')
for col in [df.columns[i] for i in range(0, len(df.columns))]:
    print(f'{col} : {np.sum(df[col].isna())}')

# %% Branch
# Undersøgelse af værdier for brugere med Nan-værdier for branch. Kan de smides ud? ja
df[df['branch'].isnull()]



# %% Undersøg aldre med Nan
# Tjek fordeling i forhold til konverteret eller ej

# Antal bruger med uregistreret alder
n_nan_age = len(df[df['age'].isna()])

# Antal bruger med uregistreret alder OG som er konverteret
n_nan_converted = len(df[(df['age'].isna()) & (df['converted']==1)])

rate_nan_convert = n_nan_converted/n_nan_age

print(f'''
Antal uregistrerede aldre er {n_nan_age}
af disse er {n_nan_converted} konverteret
hvilket giver en konverteringsratio for Nan-aldre på {rate_nan_convert:.3f}
''')

# %% Check om account er none:

acc_is_none = '9b2d5b4678781e53038e91ea5324530a03f27dc1d0e5f6c9bc9d493a23be9de0'

# Lav ny søjle med dikotomisk værdi efter om der findes konto eller ej
df['account_exists'] = df['credit_account_id'] != acc_is_none

print(f'Number of not registered accounts: {df.shape[0]-np.sum(df["account_exists"])}')

# investigate if there are only unique accounts other than 'none'
acc_names = []
for acc_name in df.credit_account_id:
    if acc_name not in acc_names:
        acc_names.append(acc_name)

print(f'\nNumber of unique account names: {len(acc_names)}')


# %% Lav ny søjle for boolean for køn
df['male'] = df['gender'] == 'male'

# %% Tjek middelværdien for konverterede brugere med henholdsvis registreret og ikke-registreret account.

df_spec = df[['account_exists','converted']]

print(f'''Middelværdien for konverterede brugere med henholdsvis registreret og ikke-registreret account:

{df.groupby("account_exists")["converted"].mean()}''')



# %% Plotfunktion for kategorisk data.
def categorical_compare_converted(category):
    """
    Funktion der plotter fordeling af konverterede og ikke-konverterede for forskellige værdier i kategorisk feature - sakset fra nettet.
    """
    fs = 25 # Sæt fontsize for plots

    pal = {1:"seagreen", 0:"gray"}
    sns.set(style="darkgrid")
    plt.subplots(figsize = (15,8))
    ax = sns.countplot(x = category, 
                    hue="converted",
                    data = df, 
                    linewidth=4, 
                    palette = pal
    )

    ## Fixing title, xlabel and ylabel
    plt.title("Fordeling af konvertering ved kategorien " + category, pad=40, fontsize=fs+10)
    plt.xlabel(category, fontsize=fs+5);
    plt.ylabel("Antal", fontsize=fs+5)

    ## Fixing xticks
    #labels = ['Female', 'Male']
    #plt.xticks(sorted(train.Sex.unique()), labels)

    ## Fixing legends
    plt.setp(ax.get_legend().get_texts(), fontsize=fs)
    plt.setp(ax.get_legend().get_title(), fontsize=fs) 
    leg = ax.get_legend()
    # leg.set_title("Converted")
    legs = leg.texts
    legs[0].set_text("Konverteret")
    legs[1].set_text("Ikke konverteret")
    plt.tick_params(labelsize=fs)
    
    plt.show()

# Plot af forskellige kategoriske data i den følgende liste
cat_list_plots = ['customer_segment', 'gender', 'account_exists', 'branch']

for category in cat_list_plots:
    categorical_compare_converted(category)


# %% Implementering af Decision tree-model for at opnå estimat for hvilke faktorer er vigtigst ved bestemmelse af konvertering

# Først skal kategorisk data omformes til kvantitativ. Dette gøres ved følgende:
#Liste over kategoriske variable
cat_list = ['converted', 'customer_segment', 'gender', 'credit_account_id', 'branch']
# Sæt type til kategorisk
for cat in cat_list:
    df[cat] = df[cat].astype('category')

# Få pandas til at omforme til numerisk
df[cat_list] = df[cat_list].apply(lambda x: x.cat.codes)


# Skalering af fee levels, da disse har stor spredning.
df['scaled_fees'] = np.log10(df['initial_fee_level']+1) # Scaling, +1 for at undgå log(0)


# Histogram kan plottes for at tjekke om værdier giver en mere rimelig (normal)fordeling

# ax = df['scaled_fees'].hist() 
# ax.set_xlabel('Log_10(fee)', fontsize=14)
# ax.set_ylabel('Count', fontsize=14)
# ax.set_title('Logaritmisk skaleret fee', fontsize=14)

# Lav en liste over kategorier der skal med i decision tree
DT_list = ['customer_segment', 'gender', 'account_exists', 'branch', 'age', 'related_customers', 'family_size', 'scaled_fees']

df_DT = df.dropna() # Smid Nan-værdier væk

# Definition af data og target for vores model
X = df_DT[DT_list]
y = df_DT['converted']

# Nedenstående er standardimplementering taget fra nettet, som finder feature importance
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X, y)
# get importance - 
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))


# plot af feature importance
fig2, ax2 = plt.subplots(dpi=150)
ax2.bar([x for x in range(len(importance))], importance)
ax2.set_xticks(range(len(DT_list)))
ax2.set_xticklabels(DT_list, rotation = 270)
ax2.set_title('Feature importance bestemt ved decision tree-algoritme', fontsize=15)
ax2.set_ylabel('Percentuel feature importance')

