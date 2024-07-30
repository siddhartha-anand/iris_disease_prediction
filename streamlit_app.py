import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title('IRIS SPECIES')
st.write('This web app predict the Iris Species of plants. ')

df = datasets.load_iris()

new_df = pd.DataFrame(df.data,columns=df.feature_names)

l = []
for x in df.target:
    if x == 0:
        l.append('setosa')
    elif x == 1:
        l.append('versicolor')
    else:
        l.append('virginica')
new_df['species'] = pd.Series(l)

X = df.data
Y = df.target

X_train , X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, stratify=Y,random_state=42)

svc_model = SVC(kernel='linear', C=1.0,random_state=42)
svc_model.fit(X_train,Y_train)

Y_pred = svc_model.predict(X_test)

cr = classification_report(Y_test,Y_pred)
ac = accuracy_score(Y_test,Y_pred)
cm = confusion_matrix(Y_test,Y_pred)

fig = sns.pairplot(new_df,hue='species');
st.pyplot(fig)
st.subheader('Heatmap')
fig1 = plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=df.target_names, yticklabels=df.target_names)
st.pyplot(fig1)