import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from io import StringIO

st.set_option('deprecation.showPyplotGlobalUse', False)

def top_5_rows(uploaded_file):
    st.write("Showing bottom 5 rows:")
    return uploaded_file.head()
def bottom_5_rows(uploaded_file):
    st.write("Showing bottom 5 rows:")
    return uploaded_file.tail()

def columns_name(uploaded_file):
    return uploaded_file.columns.tolist()

def random_rows(uploaded_file):
    st.write("Random Sample of 10 rows:")
    return uploaded_file.sample(n=10)

def a_specific_row(uploaded_file, row_rank):
    st.write("Everything about the row: **" + str(row_rank) + "**")
    specific_row = uploaded_file.loc[row_rank]
    return specific_row

def specific_rows(uploaded_file, row_rank1, row_rank2):
    st.write("Everything about the **" + str(row_rank1) + "** to **" + str(row_rank2) + "** rows")
    specific_rows = uploaded_file.iloc[row_rank1:row_rank2 + 1]
    return specific_rows

def statistics(uploaded_file, action, groupby_x, namesList):

    st.write(f"**{len(namesList)}** columns were selected: **{', '.join(namesList)}** and will be grouped by **{groupby_x}**")

    selected_columns = None  # Default initialization

    if action == "Select an action":
        st.markdown("<p class='header'>Select an action from the select box </p>", unsafe_allow_html=True)
    if action == "Mean":
        selected_columns = uploaded_file.groupby(groupby_x)[namesList].mean()
    if action == "Sum":
        selected_columns = uploaded_file.groupby(groupby_x)[namesList].sum()
    if action == "Median":
        selected_columns = uploaded_file.groupby(groupby_x)[namesList].median()
    if action == "Variance":
        selected_columns = uploaded_file.groupby(groupby_x)[namesList].var()
    if action == "Standard Deviation":
        selected_columns = uploaded_file.groupby(groupby_x)[namesList].std()
    if action == "Range":
        range_result = uploaded_file.groupby(groupby_x)[namesList].agg(lambda x: x.max() - x.min())
        selected_columns = range_result

    return selected_columns

def line_plot(uploaded_file, column1_name, column2_name):
    st.write("This plot allows us to see: **trends of a numeric variable, time series**")
    sns.set(style="whitegrid")
    # Create a line plot
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    plt.xticks(rotation=90)
    sns.lineplot(x=column1_name, y=column2_name, data=uploaded_file, marker='o',
                 color='b', label='Value', ax=ax)

    # Add labels
    plt.xlabel(column1_name)
    plt.ylabel(column2_name)
    plt.title('Line Plot')

    # Show legend
    plt.legend()

    # Plot
    st.pyplot()

def scatter_plot(uploaded_file, column1_name, column2_name):
    st.write("The **scatter plot** helps us to see **the relationship between two numerical variables and to identify correlations and clusters in the data.**")

    col1, col2 = st.columns([1, 3])  # Divide the espace in 1/4 and 3/4

    with col1:
        correlation = np.corrcoef(uploaded_file[column1_name], uploaded_file[column2_name])[0, 1]
        st.write(f"Correlation between **{column1_name}** and **{column2_name}**: **{correlation}**")

    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(x=column1_name, y=column2_name, data=uploaded_file, ax=ax, label=f'{column1_name} vs {column2_name}')
        plt.xticks(rotation=90)
        ax.legend()
        # Plot
        st.pyplot(fig)

def box_plot(uploaded_file, num_variable, cat_variable):
    st.write("**Box plot** allows us to **visualize the distribution of data and identify outliers (excluding distribution) and to compare the distributions of different datasets.**")

    # Check if column2_name exists
    if cat_variable in uploaded_file.columns:
        column_values = uploaded_file[cat_variable]

        # Check for missing values
        if column_values.isnull().any():
            # Handle missing values here or consider dropping rows with missing values
            pass
        else:
            # Ensure the data type of the column is appropriate for your operation

            if len(set(column_values)) > 10:
                st.write(f"THERE IS A LOT OF CATEGORIES IN **{cat_variable}**! IMPOSSIBLE TO PLOT")
            else:
                col1, col2 = st.columns([3, 1])  # Division of the espace in 3/4 and 1/4

                with col1:  # Use 3/4 of the space for the plot
                    fig, ax = plt.subplots()
                    sns.boxplot(x=cat_variable, y=num_variable, data=uploaded_file, ax=ax)
                    plt.xticks(rotation=90)

                    st.pyplot(fig)

                with col2:  # Use of 1/4 of the space for the table
                    st.write("Mean values after grouping:")
                    mean_values = uploaded_file.groupby(cat_variable)[num_variable].mean()
                    st.write(mean_values)
    else:
        st.write(f"Exception : There is no numerical or categorical variable on the dataframe.")

def histogram(uploaded_file, column1_name):
    st.write("The **histogram plot** allows us to **visualize the distribution of a categorical variable in relation to a numerical variable, to visualize the distribution of a 'single' variable and also to identify the dominant values (or range of values).**")
    if len(set(uploaded_file[column1_name])) > 10:
        st.write(f"THERE IS A LOT OF CATEGORIES IN **{column1_name}**! IMPOSSIBLE TO PLOT")
    else:
        fig, ax = plt.subplots()
        sns.histplot(x=column1_name, data=uploaded_file, ax=ax, color='blue')
        plt.xticks(rotation=90)
        # afficher
        st.pyplot(fig)

def kde_plot(uploaded_file, column1_name):
    st.write("The **KDE plot** allows us to **identify data modality type (bimodal, multimodal, etc.) and to compare the distribution of the same variable but for different populations.**")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 5))

    sns.kdeplot(data=uploaded_file[column1_name], ax=ax1, color='blue', fill=True)
    sns.histplot(x=column1_name, data=uploaded_file, ax=ax2, color='red', kde=True)

    # Show plots side by side
    plt.tight_layout()

    # Show the entire plot
    st.pyplot(fig)

def violin_plot(uploaded_file, num_variable, cat_variable):
    st.write("The **violin plot** is the combination of **Box plot and KDE.**")

    # Check if column2_name exists
    if cat_variable in uploaded_file.columns:
        column_values = uploaded_file[cat_variable]

        # Check for missing values
        if column_values.isnull().any():
            # Handle missing values here or consider dropping rows with missing values
            pass
        else:
            # Ensure the data type of the column is appropriate for your operation

            if len(set(column_values)) > 10:
                st.write(f"THERE IS A LOT OF CATEGORIES IN **{cat_variable}**! IMPOSSIBLE TO PLOT")

            else:
                col1, col2 = st.columns([3, 1]) # Division of the espace in 3/4 and 1/4

                with col1:  # Use 3/4 of the space for the plot
                    fig, ax = plt.subplots()

                    sns.violinplot(x=cat_variable, y=num_variable, data=uploaded_file, ax= ax, color='blue')

                    plt.xticks(rotation=90)

                    st.pyplot(fig)

                with col2:  # Use of 1/4 of the space for the table
                    st.write("Mean values after grouping:")
                    mean_values = uploaded_file.groupby(cat_variable)[num_variable].mean()
                    st.write(mean_values)
    else:
        st.write(f"Exception : There is no numerical or categorical variable on the dataframe.")

def bar_plot(uploaded_file, num_variable, cat_variable):
    st.write("The **bar plot** allows to **visualize 'multiple' categorical variables versus a numeric variable.**")

    # Check if column2_name exists
    if cat_variable in uploaded_file.columns:
        column_values = uploaded_file[cat_variable]

        # Check for missing values
        if column_values.isnull().any():
            # Handle missing values here or consider dropping rows with missing values
            pass
        else:
            # Ensure the data type of the column is appropriate for your operation

            if len(set(column_values)) > 10:
                st.write(f"THERE IS A LOT OF CATEGORIES IN **{cat_variable}**! IMPOSSIBLE TO PLOT")

            else:
                col1, col2 = st.columns([3, 1]) # Division of the espace in 3/4 and 1/4

                with col1:  # Use 3/4 of the space for the plot
                    fig, ax = plt.subplots()

                    sns.barplot(x=cat_variable, y=num_variable, data=uploaded_file, ax=ax, label=cat_variable)

                    plt.xticks(rotation=90)
                    ax.legend()

                    st.pyplot(fig)

                with col2:  # Use of 1/4 of the space for the table
                    st.write("Mean values after grouping:")
                    mean_values = uploaded_file.groupby(cat_variable)[num_variable].mean()
                    st.write(mean_values)
    else:
        st.write(f"Exception : There is no numerical or categorical variable on the dataframe.")

def heat_map(uploaded_file):
    st.write("**Heatmap** allows to see **the correlation matrix in a visual way and the concentration of values in a 2D space.**")
    dataset_correlation = uploaded_file.corr()
    fig, ax = plt.subplots()
    sns.heatmap(data=dataset_correlation, annot=True, cmap='YlGnBu', cbar=True)
    plt.title('Heatmap')
    st.write("Matrix correlation:")
    st.write(dataset_correlation)

    # afficher
    st.pyplot(fig)

def pie_chart(uploaded_file, cat_variable):
    st.write("**Pie chart** allows to see **how a variable is shared between different categories. The variable represents all (that means 100%).**")

    # Check if column2_name exists
    if cat_variable in uploaded_file.columns:
        column_values = uploaded_file[cat_variable]

        # Check for missing values
        if column_values.isnull().any():
            # Handle missing values here or consider dropping rows with missing values
            pass
        else:
            # Ensure the data type of the column is appropriate for your operation
            unique_value = uploaded_file[cat_variable].unique()
            labels = unique_value
            count_unique_value = uploaded_file[cat_variable].value_counts()
            sizes = count_unique_value

            if len(labels) >= 20:
                st.write("**THERE IS A LOT OF CATEGORIES TO SHOW! REJECTED ACTION!**")

            else:
                colors_tab = sns.color_palette('pastel')
                plt.figure(figsize=(6, 6))
                plt.pie(sizes, labels=labels, colors=colors_tab, autopct='%1.1f%%')
                plt.title('Pie Chart')
                st.pyplot()
    else:
        st.write(f"Exception : There is no categorical variable on the dataframe.")

def pair_plot(uploaded_file):
    st.write("With **pair plot** we can **visualize data distribution and relationships between numeric variables in your dataset.**")
    numerical_variables_uploaded_file = uploaded_file.select_dtypes(include=['number'])
    fig = sns.pairplot(numerical_variables_uploaded_file)
    # afficher
    st.pyplot(fig)

def linear_regression(uploaded_file, column_y, columns_x):
    x_train, x_test, y_train, y_test = train_test_split(columns_x, column_y, test_size=0.2)

    # Data standardization
    scaler = StandardScaler()
    x_train_scal = scaler.fit_transform(x_train)
    x_test_scal = scaler.transform(x_test)
    y_train_scal = scaler.fit_transform(y_train.values.reshape(-1, 1))  # Reshape y_train to ensure that it has the form we want
    y_test_scal = scaler.transform(y_test.values.reshape(-1, 1))  # Reshape y_test

    # Training
    lr = LinearRegression()
    lr.fit(x_train_scal, y_train_scal)
    y_predict = lr.predict(x_test_scal)

    # Calculate metrics
    MODEL_SCORE = lr.score(x_test_scal, y_test_scal) * 100
    MSE = mean_squared_error(y_predict, y_test_scal) * 100
    MAE = mean_absolute_error(y_predict, y_test_scal) * 100

    # Show metrics
    st.write(f"THE SCORE IS : **{MODEL_SCORE:.3f}%**")
    st.write(f"MSE : **{MSE:.3f}%**")
    st.write(f"MAE : **{MAE:.3f}%**")

    # Plot the line of the linear regression
    fig, ax = plt.subplots()
    ax.scatter(y_test_scal, y_predict, label='Data')
    ax.plot(y_test_scal, y_test_scal, color='red', label='Regression Line')
    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    ax.set_title('Linear Regression: Actual vs Predicted')
    ax.legend()

    # Show the plot
    st.pyplot(fig)

def classification_with_SVM(column_y, columns_x):
    if column_y.empty:
        st.write("Exception: There is no categorical variable in column_y.")
    else:
        st.write(columns_x)

        # Encoding of the categorical variable
        label_encoder = LabelEncoder()
        column_y = label_encoder.fit_transform(column_y)

        X = columns_x  # Features
        y = column_y  # Target

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

        # Create a SVM object
        svm_model = SVC(kernel='linear')

        # Train the model on the training data
        svm_model.fit(X_train, y_train)

        # Calculate predictions
        y_pred = svm_model.predict(X_test)

        # Model evaluation according input data
        accuracy = svm_model.score(X_test, y_test)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.write(f"**Accuracy: {accuracy * 100:.3f}%**")
        st.write("-> It's a general metric that measures the overall ability of the model to predict classes correctly.")
        st.write(f"**Precision: {precision * 100:.3f}%**")
        st.write("-> It's a measure of the model's accuracy in positive predictions.")
        st.write(f"**Recall: {recall * 100:.3f}%**")
        st.write("-> Also known as sensitivity or true positive rate, it focuses on the model's ability to identify all positive samples.")
        st.write(f"**F1 Score: {f1 * 100:.3f}%**")
        st.write("-> It's used when considering both precision and recall to evaluate the model's performance.")

        user_input = []
        for col in columns_x:
            user_value = st.number_input(f"Enter value for {col}", key=col)
            user_input.append(user_value)

        # Prediction with input data
        predicted_y = svm_model.predict([user_input])
        predicted_class = label_encoder.inverse_transform(predicted_y)[0]
        st.write("Predicted class:", predicted_class)
