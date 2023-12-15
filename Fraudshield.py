#LOADING DATASET & PROCESSING
import pandas as pd
from collections import Counter
df = pd.read_csv('credit.csv')

print(df.columns[df.isna().any()])

    #performing basic EDA
# print(df.shape)
# print(df.describe())
# print(df.info())

# print(df['fraud'].value_counts())

#SEGGREGATING & SPLITTING DATASET
# df = df.sample(frac=1)    #shuffling the dataset

    # seggreagting dataset into input and output
x = df[['repeat_retailer','used_chip','used_pin_number',
        'online_order']]#selecting input columns
x = x.iloc[:, :-1].values

print(x)
y = df.iloc[:,-1].values#selecting output column

    #since the dataset is imbalanced we will use a technique called SMOTE to balance the dataset

print(Counter(y))#checking output distribution before SMOTE
from imblearn.over_sampling import SMOTE#importing SMOTE
sm = SMOTE()#calling smote
X,Y = sm.fit_resample(x,y)#resampling using SMOTE
print(Counter(Y))#checking output distribution after SMOTE

    #splitting dataset into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=22,test_size=0.1)

    #scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#CREATING A STREAMLIT WEBPAGE
import streamlit as st
from streamlit_option_menu import option_menu

st.title('CREDIT CARD FRAUD DETECTOR')
with st.sidebar:
    option = option_menu('SELECT A MODEL',options=['LOGISTIC REGRESSION','DECISION TREE','NAIVE BAYES','KNN','MODEL GRAPHS'])

yes_no = {"YES":1,"NO":0}

card_number = st.number_input("ENTER YOUR CREDIT CARD NUMBER",max_value=999999999999)

expiry_date = st.date_input("ENTER EXPIRY DATE")

cvv = st.number_input("ENTER THREE DIGIT CVV",max_value=999)

banks_in_india = [    'Allahabad Bank',    'Andhra Bank',    'Axis Bank',    'Bank of Bahrain and Kuwait',    'Bank of Baroda',    'Bank of India',    'Bank of Maharashtra',    'Canara Bank',    'Central Bank of India',    'City Union Bank',    'Corporation Bank',    'Dena Bank',    'Deutsche Bank',    'Development Credit Bank',    'Dhanlaxmi Bank',    'Federal Bank',    'HDFC Bank',    'ICICI Bank',    'IDBI Bank',    'Indian Bank',    'Indian Overseas Bank',    'IndusInd Bank',    'Jammu and Kashmir Bank',    'Karnataka Bank Ltd',    'Karur Vysya Bank',    'Kotak Bank',    'Lakshmi Vilas Bank',    'Oriental Bank of Commerce',    'Punjab and Sind Bank',    'Punjab National Bank',    'South Indian Bank',    'Standard Chartered Bank',    'State Bank of Bikaner and Jaipur',    'State Bank of Hyderabad',    'State Bank of India',    'State Bank of Mysore',    'State Bank of Patiala',    'State Bank of Travancore',    'Syndicate Bank',    'Tamilnad Mercantile Bank Ltd',    'UCO Bank',    'Union Bank of India',    'United Bank of India',    'Vijaya Bank',    'Yes Bank']
bank_name = st.selectbox("ENTER BANK NAME",options=banks_in_india)

st.markdown("---")

# distance_from_home = st.text_input('ENTER THE DISTANCE OF TRANSACTION : ')
# distance_from_last_transaction = st.text_input('ENTER THE DISTANCE BETWEEN LAST 2 TRANSACTIONS : ')
# ratio_to_median_purchase_price= st.text_input('ENTER AMOUNT TO AVERAGE PURCHASE PRICE RATIO  : ')

repeat_retailer = st.selectbox("HAS THE CUSTOMER PREVIOUSLY MADE PAYMENT TO THIS RETAILER?",options={"YES","NO"})
repeat_retailer = yes_no[repeat_retailer]

used_chip = st.selectbox("WAS TRANSACTION MADE USING CARD TAP?",options={"YES","NO"})
used_chip = yes_no[used_chip]

used_pin_number = st.selectbox("WAS CREDIT CARD PIN USED?",options={"YES","NO"})
used_pin_number = yes_no[used_pin_number]

online_order = st.selectbox("IS THE TRANSACTION ONLINE?",options={"YES","NO"})
online_order = yes_no[online_order]

submit = st.button('SUBMIT')

st.markdown("---")

new = [[repeat_retailer,used_chip,used_pin_number,
        online_order]]

#TRAINING THE ML MODELS & INTEGRATING WITH WEBPAGE
if submit:
        # if card_number or cvv == 0:
        #     st.error("CHECK ALL INPUTS AND CLICK SUBMIT")
        # else:
    if option == 'LOGISTIC REGRESSION':
        # LOGISTIC REGRESSION
        from sklearn.linear_model import LogisticRegression  # importing the package

        logistic_regression = LogisticRegression()  # calling the function
        logistic_regression.fit(x_train, y_train)  # training the model
        logistic_regression_test_prediction = logistic_regression.predict(
            x_test)  # using the trained model to predict for x_test
        logistic_regression_new_prediction = logistic_regression.predict(
            sc.transform(new))  # using the model to predict for new user input
        # DISPLAYING MODEL RESULTS
        st.header("RESULT OF LOGISTIC REGRESSION MODEL IS : ")
        if logistic_regression_new_prediction[0] == 1:
            st.error("THERE IS A POTENTIAL CREDIT CARD FRAUD")
        else:
            st.success("NO CREDIT CARD FRAUD DETECTED")
            # DISPLAYING MODEL PARAMETERS
        st.header("MODEL PARAMETERS : ")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # importing metrics

        st.info("accuracy of the model : {0}%".format(
            accuracy_score(y_test, logistic_regression_test_prediction) * 100))
        st.error("precision of the model : {0}%".format(
            precision_score(y_test, logistic_regression_test_prediction) * 100))
        st.warning("recall score of the model : {0}%".format(
            recall_score(y_test, logistic_regression_test_prediction) * 100))
        st.success(
            "f1 score of the model : {0}%".format(f1_score(y_test, logistic_regression_test_prediction) * 100))
        # DISPLAYING CLASSIFICATION REPORT
        from sklearn.metrics import classification_report  # importing

        st.header('CLASSIFICATION REPORT')
        report = classification_report(y_test, logistic_regression_test_prediction,
                                       output_dict=True)  # creating classification report
        report_df = pd.DataFrame(report).transpose()  # converting the report into a dataframe
        st.dataframe(report_df, height=212, width=1000)  # displaying the dataframe on the webpage

    if option == 'DECISION TREE':
        # DECISION TREE
        from sklearn.tree import DecisionTreeClassifier

        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(x_train, y_train)
        decision_tree_test_prediction = decision_tree.predict(x_test)
        decision_tree_new_prediction = decision_tree.predict(sc.transform(new))
        # DISPLAYING MODEL RESULTS
        st.header("RESULT OF DECISION TREE MODEL IS : ")
        if decision_tree_new_prediction[0] == 1:
            st.error("THERE IS A POTENTIAL CREDIT CARD FRAUD")
        else:
            st.success("NO CREDIT CARD FRAUD DETECTED")
            # DISPLAYING MODEL PARAMETERS
        st.header("MODEL PARAMETERS : ")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # importing metrics

        st.info("accuracy of the model : {0}%".format(accuracy_score(y_test, decision_tree_test_prediction) * 100))
        st.error(
            "precision of the model : {0}%".format(precision_score(y_test, decision_tree_test_prediction) * 100))
        st.warning(
            "recall score of the model : {0}%".format(recall_score(y_test, decision_tree_test_prediction) * 100))
        st.success("f1 score of the model : {0}%".format(f1_score(y_test, decision_tree_test_prediction) * 100))
        # DISPLAYING CLASSIFICATION REPORT
        from sklearn.metrics import classification_report  # importing

        st.header('CLASSIFICATION REPORT')
        report = classification_report(y_test, decision_tree_test_prediction,
                                       output_dict=True)  # creating classification report
        report_df = pd.DataFrame(report).transpose()  # converting the report into a dataframe
        st.dataframe(report_df, height=212, width=1000)  # displaying the dataframe on the webpage

    if option == 'NAIVE BAYES':
        # RANDOM FOREST
        from sklearn.naive_bayes import GaussianNB

        naive_bayes = GaussianNB()
        naive_bayes.fit(x_train, y_train)
        naive_bayes_test_prediction = naive_bayes.predict(x_test)
        naive_bayes_new_prediction = naive_bayes.predict(sc.transform(new))
        # DISPLAYING MODEL RESULTS
        st.header("RESULT OF NAIVE BAYES MODEL IS : ")
        if naive_bayes_new_prediction[0] == 1:
            st.error("THERE IS A POTENTIAL CREDIT CARD FRAUD")
        else:
            st.success("NO CREDIT CARD FRAUD DETECTED")
            # DISPLAYING MODEL PARAMETERS
        st.header("MODEL PARAMETERS : ")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # importing metrics

        st.info("accuracy of the model : {0}%".format(accuracy_score(y_test, naive_bayes_test_prediction) * 100))
        st.error("precision of the model : {0}%".format(precision_score(y_test, naive_bayes_test_prediction) * 100))
        st.warning(
            "recall score of the model : {0}%".format(recall_score(y_test, naive_bayes_test_prediction) * 100))
        st.success("f1 score of the model : {0}%".format(f1_score(y_test, naive_bayes_test_prediction) * 100))
        # DISPLAYING CLASSIFICATION REPORT
        from sklearn.metrics import classification_report  # importing

        st.header('CLASSIFICATION REPORT')
        report = classification_report(y_test, naive_bayes_test_prediction,
                                       output_dict=True)  # creating classification report
        report_df = pd.DataFrame(report).transpose()  # converting the report into a dataframe
        st.dataframe(report_df, height=212, width=1000)  # displaying the dataframe on the webpage

    if option == 'KNN':
        # RANDOM FOREST
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier()
        knn.fit(x_train, y_train)
        knn_test_prediction = knn.predict(x_test)
        knn_new_prediction = knn.predict(sc.transform(new))
        # DISPLAYING MODEL RESULTS
        st.header("RESULT OF KNN MODEL IS : ")
        if knn_new_prediction[0] == 1:
            st.error("THERE IS A POTENTIAL CREDIT CARD FRAUD")
        else:
            st.success("NO CREDIT CARD FRAUD DETECTED")
            # DISPLAYING MODEL PARAMETERS
        st.header("MODEL PARAMETERS : ")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # importing metrics

        st.info("accuracy of the model : {0}%".format(accuracy_score(y_test, knn_test_prediction) * 100))
        st.error("precision of the model : {0}%".format(precision_score(y_test, knn_test_prediction) * 100))
        st.warning("recall score of the model : {0}%".format(recall_score(y_test, knn_test_prediction) * 100))
        st.success("f1 score of the model : {0}%".format(f1_score(y_test, knn_test_prediction) * 100))
        # DISPLAYING CLASSIFICATION REPORT
        from sklearn.metrics import classification_report  # importing

        st.header('CLASSIFICATION REPORT')
        report = classification_report(y_test, knn_test_prediction,
                                       output_dict=True)  # creating classification report
        report_df = pd.DataFrame(report).transpose()  # converting the report into a dataframe
        st.dataframe(report_df, height=212, width=1000)  # displaying the dataframe on the webpage

    if option == 'MODEL GRAPHS':
        # DISPLAYING MODEL GRAPHS
        st.header("MODEL GRAPHS")
        import matplotlib.pyplot as plt

        accuracies = [71.35, 70.27, 74.05, 72.97]
        precisions = [70.535, 71.15, 68.08, 72.47]
        models = ['LOGI REG', 'DECISION TREE', 'NAIVE BAYES', 'KNN']

        # Plot models vs accuracies
        fig, ax = plt.subplots()
        ax.bar(models, accuracies,color='red')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy comparison')
        st.pyplot(fig)

        # Plot models vs precisions
        fig, ax = plt.subplots()
        ax.bar(models, precisions)
        ax.set_ylabel('Precision')
        ax.set_title('Precision Comparison')
        st.pyplot(fig)

        #best model declarations
        st.subheader("Based on these results, we can see that the model with the highest accuracy is DECISION TREE, and the model with the highest precision is KNN. However, if we consider both metrics together, we can see that the model with the highest average accuracy and precision is LOGISTIC REGRESSION, making it the best model overall according to the given metrics.")
else:
    st.error("CLICK SUBMIT")