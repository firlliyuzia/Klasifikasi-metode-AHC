import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pandas import DataFrame
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas import DataFrame
from sklearn import metrics


st.header("Clustering Dataset Golongan Darah Menggunakan Algoritma AHC")

selected = option_menu(
    menu_title = None, #wajib ada
    options=["Dataset", "Prepocessing", "Model"],
    icons=["book-half","cast", "briefcase-fill"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container":{"padding":"0!important", "background-color":"#ffffff",},
        "icons":{"font-size":"14px"},
        "nav-link":{"font-size":"15px",
            "text-align":"center",
            "margin":"0px",
            "--hover-color":"#eee",
        },
    }
)

def load_data():
    pd_crs = pd.read_csv("data.csv")
    return pd_crs

# Hanya akan di run sekali
pd_crs = load_data()

def processing():
    del(pd_crs['NO'])

    # label_encoder object
    label_encoder = preprocessing.LabelEncoder()
    # Encode labels in column 'gender'.
    pd_crs['JENIS KELAMIN']= label_encoder.fit_transform(pd_crs['JENIS KELAMIN'])
    pd_crs['JENIS KELAMIN'].unique()

    tmp_lhr=pd.get_dummies(pd_crs['TEMPAT LAHIR'])
    tensi=pd.get_dummies(pd_crs['TENSI'])
    goldar=pd.get_dummies(pd_crs['GOL.DAR'])

    scaler = MinMaxScaler()
    numerik = pd.DataFrame(pd_crs, columns=['HB', 'Umur', 'BB (Kg)'])
    scaler.fit(numerik)
    crs_new = scaler.transform(numerik)

    crs_new = DataFrame(crs_new)
    del(pd_crs['HB'], pd_crs['Umur'], pd_crs['BB (Kg)'], pd_crs['TENSI'], pd_crs['TEMPAT LAHIR'], pd_crs['GOL.DAR'])

    pd_crs_new = pd.concat([pd_crs, crs_new, tmp_lhr, tensi, goldar], axis=1)

    pd_crs_new.rename(columns = {0: "HB", 1: "Umur", 2: "BB (kg)"}, inplace=True)

    return pd_crs_new

global data_cluster

if selected == "Dataset":
    st.write('''#### Dataset''')
    st.write(pd_crs)
    st.write("""
    Data yang akan dianalisis adalah data tentang golongan darah, yang diambil dari data pendonor di PMI Kabupaten Bnagkalan.
    """)

    st.write('''#### Fitur-fitur pada dataset''')
    st.write("Pada dataset ini terdiri sebanyak 100 data dengan 8 fitur. Adapun fitur-fiturnya yaitu:")
    st.info('''
    1. NO untuk nomor urut pendonor
    2. JENIS KELAMIN (Jenis kelamin pendonor (P/L))
    3. HB (Protein yang ada di dalam sel darah merah)
    4. BB (Kg) (Berat badan pendonor dalam kilogram)
    5. UMUR (Umur pendonor)
    6. TENSI (Tekanan darah pendonor)
    7. TEMPAT LAHIR (Tempat lahir pendonor )
    8. GOL.DAR (Golongan darah pendonor (A, B, AB, O))
    ''')


    st.write('\n')
    st.write('\n')
    st.write("Source Code : ")

if selected == "Prepocessing":

    st.write("#### Data sebelum normalisasi")
    st.write(pd_crs.iloc[:5])

    pd_crs_new = processing()
    st.write("#### Data setelah normalisasi")
    st.write(pd_crs_new.iloc[:5])


if selected == "Model":
    pd_crs_new = processing()

    st.write("### Algoritma AHC (Agglomerative Hierarchical Clustering)")
    st.write("")
    st.write("#### Dendogram Data")

    fig = plt.figure(figsize = (50,20))
    Z = linkage(pd_crs_new, optimal_ordering = False, method = 'ward')
    dn = dendrogram(Z, truncate_mode = 'lastp')
    st.pyplot(fig)

    
    X = pd_crs_new.iloc[:, [0, 41]].values
    model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')
    model.fit(X)
    labels = model.labels_

    model1 = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='single')
    model1.fit(X)
    label1 = model1.labels_

    model2 = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single')
    model2.fit(X)
    label2 = model2.labels_

    data_donor = load_data()
    label=pd.DataFrame(labels, columns=['label'])
    data_cluster = pd.concat([data_donor, label], axis=1)

    st.write("")
    st.write("#### Data Label")
    st.write(data_cluster)

    eval1 = metrics.silhouette_score(X, labels, metric='euclidean')
    st.write("")
    st.write("#### Evaluasi Cluster sebanyak 3")
    st.write(eval1)

    eval2 = metrics.silhouette_score(X, label1, metric='euclidean')
    st.write("")
    st.write("#### Evaluasi Cluster sebanyak 4")
    st.write(eval2)

    eval3 = metrics.silhouette_score(X, label2, metric='euclidean')
    st.write("")
    st.write("#### Evaluasi Cluster sebanyak 5")
    st.write(eval3)
    

# if selected == "Prediksi":

#     data2 = pd.read_csv("data1.csv")
#     def prepo(data_gol):
#         data_label = pd.DataFrame(data_gol, columns = ['LABEL'])
#         del(data_gol['NO'], data_gol['TEMPAT LAHIR'], data_gol['LABEL'])
#         # label_encoder object
#         label_encoder = preprocessing.LabelEncoder()
#         # Encode labels in column 'gender'.
#         data_gol['JENIS KELAMIN']= label_encoder.fit_transform(data_gol['JENIS KELAMIN'])
#         data_gol['JENIS KELAMIN'].unique()

#         tensi=pd.get_dummies(data_gol['TENSI'])
#         goldar=pd.get_dummies(data_gol['GOL.DAR'])

#         del(data_gol['TENSI'], data_gol['GOL.DAR'])

#         data_gol_new = pd.concat([data_gol, goldar, tensi, data_label], axis=1)

#         data_gol_new.rename(columns = {0: "HB", 1: "Umur", 2: "BB (kg)"}, inplace=True)
#         return data_gol_new

#     _, _, col3, _, _ = st.columns([1, 1, 1,1,1])
#     with col3:
#         st.write('### Input Data')

#     with st.form(key='my_form'):
#         col0, col1 = st.columns([1,1])
#         with col0:
#             jk = st.radio('Pilih Jenis Kelamin', ('Laki-laki', 'Perempuan'))
#             if jk == 'Laki-laki':
#                 jenis = 1
#             else:
#                 jenis = 0
            
#             hb = st.number_input('HB', 0)
#             umur = st.number_input('Umur', 0)
            
#         with col1:
#             bb = st.number_input('Berat Badan', 0)
#             goldar = st.text_input('Golongan Darah')
#             if goldar == 'A':
#                 gola = 1
#                 golab = 0
#                 golb = 0
#                 golo = 0
#             elif goldar =='B':
#                 gola = 0
#                 golab = 0
#                 golb = 1
#                 golo = 0
#             elif goldar =='AB':
#                 gola = 0
#                 golab = 1
#                 golb = 0
#                 golo = 0
#             else:
#                 gola = 0
#                 golab = 0
#                 golb = 0
#                 golo = 1

#             tensi = st.text_input('Tensi')
#             if tensi == '110/70':
#                 ten = 1
#                 ten1 = 0
#                 ten2= 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '110/80':
#                 ten = 0
#                 ten1 = 1
#                 ten2 = 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '120/80':
#                 ten = 0
#                 ten1 = 0
#                 ten2 = 1
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '130/80':
#                 ten = 0
#                 ten1 = 0
#                 ten2 = 0
#                 ten3 = 1
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '140/80':
#                 ten = 0
#                 ten1 = 0
#                 ten2= 0
#                 ten3 = 0
#                 ten4 = 1
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '140/90':
#                 ten = 0
#                 ten1 = 0
#                 ten2= 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 1
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '150/80':
#                 ten = 0
#                 ten1 = 0
#                 ten2= 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 1
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '150/90':
#                 ten = 0
#                 ten1 = 0
#                 ten2= 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 1
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '160/90':
#                 ten = 0
#                 ten1 = 0
#                 ten2= 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 1
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '90/70':
#                 ten = 0
#                 ten1 = 0
#                 ten2 = 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 1
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '80/70':
#                 ten = 0
#                 ten1 = 0
#                 ten2 = 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 1
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '70/70':
#                 ten = 0
#                 ten1 = 0
#                 ten2 = 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 1
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '90/80':
#                 ten = 0
#                 ten1 = 0
#                 ten2 = 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 1
#                 ten14 = 0
#                 ten15 = 0
#             elif tensi == '100/80':
#                 ten = 0
#                 ten1 = 0
#                 ten2 = 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 1
#                 ten15 = 0
#             elif tensi == '100/70':
#                 ten = 0
#                 ten1 = 0
#                 ten2 = 0
#                 ten3 = 0
#                 ten4 = 0
#                 ten5 = 0
#                 ten6 = 0
#                 ten7 = 0
#                 ten8 = 0
#                 ten9 = 0
#                 ten10 = 0
#                 ten11 = 0
#                 ten12 = 0
#                 ten13 = 0
#                 ten14 = 0
#                 ten15 = 1

#         if st.form_submit_button('Check'):
#             data = {'JENIS KELAMIN':jenis, 'HB':hb, 'Umur':umur, 'BB(Kg)':bb, 'A':gola,'B':golb, 
#             'AB':golab, 'O':golo,'110/70':ten, '110/80':ten1, '120/80':ten2, '130/80':ten3,'140/80':ten4, 
#             '140/90':ten5, '150/80':ten6, '150/90':ten7, '160/80':ten8, '160/90':ten9, '80/70':ten10, 
#             '90/70':ten11, '70/70':ten12, '90/80':ten13, '100/80':ten14, '100/70':ten15}
#             pre = pd.DataFrame(data, index=[0])

#             data_asli = {'JENIS KELAMIN':jk, 'HB':hb, 'Umur':umur, 'BB(Kg)':bb, 'GOL.DAR':goldar, 'TENSI':tensi}
#             masukan = pd.DataFrame(data_asli, index=[0])

#             data_gol_new = prepo(data2)
#             #memisahkan fitur dan label
#             feature=data_gol_new.iloc[:,0:24].values
#             label=data_gol_new.iloc[:,24].values

#             #membagi data training dan testing
#             X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=1)

#             #klasifikasi menggunakan decision tree
#             clf = tree.DecisionTreeClassifier(random_state=3, max_depth=1)
#             clf = clf.fit(X_train, y_train)
#             pred = clf.predict(X_test)
#             akurasi = accuracy_score(y_test, pred)
#             prediction = clf.predict(pre)

#             st.write(" ")
#             st.write('''#### Prediksi dengan Metode Decisison Tree''')
#             st.write("Akurasi Model Decision Tree")
#             st.info(akurasi)

#             st.write("Data yang anda masukkan")
#             st.write(masukan)

#             st.write("Kelayakan menjadi pendonor")
#             st.write(prediction)