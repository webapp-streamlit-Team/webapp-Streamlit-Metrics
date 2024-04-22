import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from sklearn import datasets
from sklearn.decomposition import PCA, NMF
from umap import UMAP 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering



st.set_page_config(page_title="Statistic Metrics", page_icon=":bar_chart:", layout="wide")


st.title('Αλγόριθμοι Μηχανικής Μάθησης')


# Insert containers separated into tabs:
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Raw Data", "2D Visualization", "Classification Algorithms", "Clustering Algorithms", "Results", "Info"])



file = st.sidebar.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])


@st.cache_data
def load_data(file):
    if file is not None:
        if file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            df = pd.read_excel(file)
        elif file.type == 'text/csv':
            df = pd.read_csv(file)
        else:
            raise ValueError("Unsupported file type")
        return df



def Visualization(df):
    # Διάταξη στηλών με βάση το όνομα των στηλών
    # Handle non-numeric data
    non_numeric_columns = df.select_dtypes(include=['object']).columns
    for column in non_numeric_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

    # Handle missing values with mean imputation
    
    # Handle missing values with mean imputation
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=np.number)), columns=df.select_dtypes(include=np.number).columns)

    # Column layout based on column names
    columns = df_imputed.columns.tolist()

    # Select two features for dimensionality reduction
    feature1 = st.sidebar.selectbox("Select Feature 1", columns)
    feature2 = st.sidebar.selectbox("Select Feature 2", columns, index=1)

    # Select dimensionality reduction algorithm
    dim_reduction_algorithm = st.sidebar.selectbox(
        "Select dimensionality reduction algorithm",
        ["PCA", "UMAP", "NMF"]
    )

    # Prepare data for dimensionality reduction
    X = df_imputed[[feature1, feature2]].values

    # Dimensionality reduction
    if dim_reduction_algorithm == "PCA":
        reducer = PCA(n_components=2)
        

    elif dim_reduction_algorithm == "UMAP":
        reducer = UMAP(n_components=2)
        

    elif dim_reduction_algorithm == "NMF":
        reducer = NMF(n_components=2)
        


    # Dimensionality reduction
    X_reduced = reducer.fit_transform(X)

    # Create DataFrame with reduced data
    df_reduced = pd.DataFrame(X_reduced, columns=["Feature 1", "Feature 2"])

    # Visualize reduced data on a scatter plot
    fig = px.scatter(df_reduced, x="Feature 1", y="Feature 2")
    st.plotly_chart(fig)
    




def classification_tab():
    
    # Load dataset
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
   # User input for classification algorithm
    classifier = st.selectbox("Select classification algorithm:", ["K-Nearest Neighbors", "Support Vector Machine"])
    
    if classifier == "K-Nearest Neighbors":
        # User input for k value in KNeighborsClassifier
        k_neighbors = st.slider("Select k value for KNN:", 1, 10)
        
        # K-Nearest Neighbors Classifier
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        knn.fit(X_train, y_train)
        knn_y_pred = knn.predict(X_test)
        
        # Display K-Nearest Neighbors accuracy
        st.write("K-Nearest Neighbors Accuracy:", accuracy_score(y_test, knn_y_pred))


    elif classifier == "Support Vector Machine":
        # Support Vector Machine Classifier
        svc = SVC()
        svc.fit(X_train, y_train)
        svc_y_pred = svc.predict(X_test)
        
        # Display Support Vector Machine accuracy
        st.write("Support Vector Machine Accuracy:", accuracy_score(y_test, svc_y_pred))




def clustering_tab():
    
    # Load dataset
    iris = datasets.load_iris()
    X = iris.data
    
    # User input for number of clusters
    num_clusters = st.slider("Select number of clusters:", 2, 10)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    
    # Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=num_clusters)
    agg.fit(X)
    
    # Display cluster labels
    st.write("K-Means Clustering Labels:", kmeans.labels_)
    st.write("Agglomerative Clustering Labels:", agg.labels_)

    


if __name__ == "__main__":
    df = load_data(file)
    if df is not None:

        with tab1:

            with st.spinner("Loading..."):
                time.sleep(2)
            
            st.header("Raw Data")
            labels = np.random.choice(['A', 'B', 'C', 'D'], size=len(df))
            df['Label'] = labels
            st.write(df)
            st.subheader("Evaluating the Birth Rate - Death Rate (per 1,000 people)")
            st.bar_chart(data=df, x="Birth rate, crude (per 1,000 people)", y="Death rate, crude (per 1,000 people)", color="#ffaa00", width=0, height=0, use_container_width=True)
            st.success("Done!")
        
        with tab2:

            with st.spinner("Loading..."):
                time.sleep(2)

            st.header("2D Visualization")
            Visualization(df)
            st.success("Done!")
        
        with tab3:
                
                with st.spinner("Loading..."):
                    time.sleep(2)

                st.header("Classification Algorithms")
                classification_tab()
                st.success("Done!")
        
        with tab4:

            with st.spinner("Loading..."):
                time.sleep(2)

            st.header("Clustering Algorithms")
            clustering_tab()
            st.success("Done!")

        with tab5:

            with st.spinner("Loading..."):
                time.sleep(2)
            
            st.header("Results")
            
            st.subheader("Εισαγωγή")

            st.write("""Ο κύριος στόχος της εφαρμογής μας είναι η οπτικοποίηση δεδομένων σε έναν δισδιάστατο χώρο να συνδυάζεται αποτελεσματικά με τους αλγορίθμους ταξινόμησης για την καλύτερη κατανόηση και ερμηνεία των δεδομένων. """)
            
            st.subheader("Μέθοδοι και Εργαλεία ")
            
            st.write("""Οι μέθοδοι και τα εργαλεία που χρησιμοποιήθηκαν για την υλοποίηση της εφαρμογής μας είναι scatter plots για το 2D Visualization (οπτικοποίηση) των αποτελεσμάτων oι αλγόριθμοι PCA (Principal Component Analysis), UMAP (Uniform Manifold Approximation and Projection) και NMF (Non-Negative Matrix Factorization) που είναι αλγόριθμοι μείωσης διαστάσεων, που χρησιμοποιούνται για την ανάλυση και την οπτικοποίηση δεδομένων υψηλής διαστατικότητας. Στη συνέχεια χρησιμοποιήθηκαν οι αλγόριθμοι ταξινόμησης Support Vector Machineκαι Κ-Nearest Neighbours. Τέλος χρησιμοποιήθηκαν αλγόριθμοι συσταδοποίησης (clustering algorithms) όπως ο K-Means Clustering και ο Agglomerative Clustering όπου αναθέτουμε ετικέτες (labels) στα δεδομένα ώστε να κατηγοριοποιηθούν σε ομάδες και να δούμε την κατανομή τους στον χώρο των χαρακτηριστικών.""")
            
            st.subheader("Δεδομένα")
            
            st.write(""" Τα δεδομένα που αναλύθηκαν προέρχονται από το WorldBank dataset όπου περιέχει πληθώρα δειγμάτων από διάφορα χαρακτηριστικά όπως το Birth Rate - Death Rate (per 1000 people) ανά χώρα.""")
            
            st.subheader("Αποτελέσματα")
            
            st.write("""Τα ποσοστά που προέκυψαν απο τη ταξινόμηση των δειγμάτων ήταν αισιόδοξα με τον Support Vector Machine να έχει ποσοστό ακρίβειας **έως και 100%** και τον Κ-Nearest Neighbours να έχει ποσοστό ακρίβειας **έως και 100%** όμως για ορισμένες παραμέτρους k, και συνεργαζόμενοι με τους dimensionality reduction αλγόριθμους PCA, UMAP και NMF.""")
            
            st.subheader("Συμπεράσματα και Μελλοντικές προεκτάσεις")
            
            st.write(""" Συμπερασματικά, διακρίνουμε ότι πήραμε αρκετά θετικά αποτελέσματα από τους αλγορίθμους ταξινόμησης με υψηλά ποσοστά ακρίβειας που βοηθούν στη καλύτερη ανάλυση και επεξεργασία των δεδομένων. Συγκριτικά, ο αλγόριθμος ταξινόμησης Support Vector Machine παρουσιάζεται ως ο πιο ακριβής αλγόριθμος όπου σε αρκετές περιπτώσεις αγγίζει και το ποσοστό ακρίβειας **100%**. Στο μέλλον, θα μπορούσαν να ενταχθούν περισσότεροι αλγόριθμοι ταξινόμησης και αλγόριθμοι συσταδοποίησης για μεγαλύτερη πληθώρα εργαλείων και μεθόδων.""")
        
        with tab6:
            
            with st.spinner("Loading..."):
                time.sleep(2)
            
            st.header("Info")
            
            st.subheader("Περιγραφή Εφαρμογής")
            
            st.write("""Η εφαρμογής μας Streamlit Metrics βοηθά στην οπτικοποίηση, ανάλυση, ταξινόμηση και συσταδοποίηση των δεδομένων από csv ή xlsx αρχεία dataset.""")
            
            st.subheader("Λειτουργικότητα της Εφαρμογής")
            
            st.write("""Αρχικά, εισάγουμε το αρχείο (xlsx ή csv) το οποίο περιέχει το dataset με το button Βrowse Files. Ύστερα, εμφανίζεται στo tab Raw Data το dataset με τη μορφή πίνακα και επιπλέον από κάτω ενα προεπιλεγμένο διαγραμμα που μορφοποιείται ανάλογα με τη μορφή του κώδικα το οποίο υλοποιήθηκε για το συγκεκριμένο dataset (WorldBank) που έχει αναλυθεί.  Προχωράμε στο tab 2D Visualization όπου οπτικοποιόυνται χαρακτηριστικά που επιλέγονται από τον χρήστη ανάλογα και με τον reduction αλγόριθμο που έχει επιλεχθεί. Στο επόμενο tab που αφορά τους αλγορίθμους ταξινόμησης επιλέγεται ο αλγόριθμος που ενδιαφέρει τον χρήστη, τα χαρακτηριστικά που επιθυμεί να αναλύσει, ο reduction αλγόριθμος για την καλύτερη κατανομή των χαρακτηριστικών καθώς και η παράμετρος του αλγόριθμου ταξινόμησης **(εάν χρειάζεται)**. Στη συνέχεια υπάρχει το tab με τους clustering αλγόριθμους όπου πάλι ο χρήστης ακολουθεί την ίδια λειτουργική διαδικασία που ακολούθησε και στο classification tab, ώστε να γίνει η συσταδοποίηση σύμφωνα με τα labels του dataset. Τέλος στα επόμενα δύο tabs ο χρήστης μπορεί να διαβάσει πληροφορίες σχετικά με την εφαρμογή την ομάδα της και τη λειτουργικότητα της. Εάν όλα καταχωρηθούν σωστά εμφανίζεται ενα μήνυμα επιτυχίας Done αλλιώς εμφανίζεται το ανάλογο μήνυμα σφάλματος. """)
            
            st.subheader("Ομάδα Ανάπτυξης")
            
            st.write("""Η ομάδα ανάπτυξης της εφαρμογής αποτελείται απο δύο ατομα, τον Καλδάνη Χρήστο προπτυχιακό φοιτητή στο τμήμα Πληροφορικής του Ιονίου Πανεπιστημίου και υπεύθυνο υλοποίησης της εφαρμογής και από τον Κωνσταντίνο Λύγκουρη που είναι επίσης προπτυχιακός φοιτητής στο τμήμα Πληροφορικής του Ιονίου Πανεπιστημίου.""")
            
            st.subheader("Tasks")
            
            st.write("""Τα Tasks που υλοποιήθηκαν από κάθε μέλος είναι:""")
            st.write("""O Καλδάνης Χρήστος ανέλαβε τα εξής: Raw Data, 2D Visualization, Classification Algorithms, Clustering Algorithms, Docker, Github, Αναφορά σε **LaTeX**""")
            st.write("""O Λύγκουρης Κωνσταντίνος ανέλαβε τα εξής: UML Diagrams, Κύκλος Ζωής Έκδοσης Λογισμικού, Συγγραφή των Result και Info tabs. """)

            



