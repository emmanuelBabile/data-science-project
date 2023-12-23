from audio_processing import *
from image_processing import *
from table_processing import *
from text_processing import *

from scipy.io.wavfile import read
from cnn import *
from keras.models import model_from_json


# Function to process uploaded audio
def process_audio(audio):
    # Perform audio processing here
    st.audio(audio, format='audio/wav')

def main():
    st.markdown(
        """
        <style>
            .page {
                background-color: #555555;
                padding: 1rem;
                color: white;
                font-size: 2rem;
                text-align: center;
            }
            .header {
                margin-bottom: 1.5rem;
                text-align: center;
            }
            .h {
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='header'>Data Science App</h1>", unsafe_allow_html=True)

    # Home page with buttons
    page = st.selectbox("Select a page", ["Home", "Audio", "Image", "Table",  "Text", "CNN image classifier"])

    if page == "Home":
        st.markdown("<h1 class='header'>Welcome to our application!</h1>", unsafe_allow_html=True)
        st.markdown("<p class='header'>Select a page from the header to upload and process different types of data.</p>"
                    , unsafe_allow_html=True)

    elif page == "Audio":
        st.markdown("<h2 class='h'>Audio Page</h2>", unsafe_allow_html=True)
        st.write("<p class='header'>Upload an audio file for processing.</p>", unsafe_allow_html=True)
        uploaded_audio = st.file_uploader("Choose an audio file", type=["wav"])
        if uploaded_audio is not None:
            # Read the uploaded audio file
            fs, data = read(uploaded_audio)

            # Extract the first 20 seconds of the data
            fs_start = int(fs * 0)  # Starting at 0 seconds
            fs_end = int(fs * 20)  # Ending at 20 seconds

            dt = data[fs_start:fs_end]
            env = []
            length = len(dt) // 2000

            for i in range(length):
                try:
                    m = np.max(dt[i * 2000: (i + 1) * 2000])
                    env.append(m)
                except:
                    pass

            env = np.array(env)

            # Display audio information
            st.write("Sampling Rate (fs):", fs)
            st.write("Number of Samples:", len(data))

            # Process the uploaded audio
            process_audio(uploaded_audio)

            audio_shape = get_audio_shape(data)
            st.write("Audio Shape:   ", audio_shape)

            # Define the options for the selectbox
            selected_option = st.selectbox("Select an action", [
                "Plot by Time",
                "Plot by Sample",
                "Detect Envelope",
                "Smooth Signal",
                "Calculate Respiration Cycles"
            ])

            # Check the selected option and perform the corresponding action
            if selected_option == "Plot by Time":
                plot_by_time(data, fs)

            if selected_option == "Plot by Sample":
                plot_first_20_seconds(data, dt, fs)

            if selected_option == "Detect Envelope":
                # Call the detect_envelope function
                env = detect_envelope(env)

            if selected_option == "Smooth Signal":
                if 'env' in locals():
                    # Check if 'env' is defined (i.e., Detect Envelope has been called)
                    box_pts = 6
                    sm_env = smooth_signal(env, box_pts)
                    # Plot the original signal
                    plt.plot(sm_env)
                    threshold = 160
                    plt.hlines(threshold, 0, 80, colors='red')
                    # Show the plot
                    st.pyplot(plt)
                else:
                    st.warning("Please run 'Detect Envelope' first to obtain the envelope signal.")

            if selected_option == "Calculate Respiration Cycles":
                threshold_cycles = 160
                sm_env = smooth_signal(env, 6)
                xs = calculate_respiration_cycles(sm_env, threshold_cycles)
                plt.plot(sm_env)
                threshold = 160
                plt.vlines(xs, 0, 200, colors='red')
                plt.hlines(threshold, 0, 80, colors='green')
                # Show the plot
                st.pyplot(plt)
                # Display the results
                st.write(f"Detected Respiration Cycles: {len(xs) // 2}")
                st.write(f"Time between Cycles: {np.diff(xs).mean()} samples")

    elif page == "Image":
        st.markdown("<h2 class='h'>Image Page</h2>", unsafe_allow_html=True)
        st.write("<p class='header'>Upload an image for processing.</p>", unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            # Open the image using OpenCV
            image_bytes = uploaded_image.read()
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)

            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Display the uploaded image
            st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

            # Functions
            function = st.selectbox("Select a function",
                                    ["Select a function", "Convert to Grayscale", "Apply Gaussian Blur",
                                     "Apply Edge Detection", "Dilate Edges", "Crop Image",
                                     "Translate Image", "Rotate Image", "Resize Image", "Flip Image",
                                     "Convolution Filter", "Detect Faces"])

            if function == "Select a function":
                st.markdown("<p class='header'>Select a function from the select box </p>", unsafe_allow_html=True)

            if function == "Convert to Grayscale":
                converted_image = convert_to_grayscale(img_rgb)
                st.image(converted_image, caption='Grayscale Image', use_column_width=True)

            if function == "Apply Gaussian Blur":
                blurred_image = apply_gaussian_blur(img_rgb)
                st.image(blurred_image, caption='Blurred Image', use_column_width=True)

            if function == "Apply Edge Detection":
                edge_detected_image = apply_edge_detection(img_rgb)
                st.image(edge_detected_image, caption='Edge Detection', use_column_width=True)

            if function == "Dilate Edges":
                dilated_image = dilate_edges(img_rgb)
                st.image(dilated_image, caption='Dilated Edges', use_column_width=True)

            if function == "Crop Image":
                cropped_image = crop_image(img_rgb)
                st.image(cropped_image, caption='Cropped Image', use_column_width=True)

            if function == "Translate Image":
                translated_image = translate_image(img_rgb)
                st.image(translated_image, caption='Translated Image', use_column_width=True)

            if function == "Rotate Image":
                rotated_image = rotate_image(img_rgb)
                st.image(rotated_image, caption='Rotated Image', use_column_width=True)

            if function == "Resize Image":
                resized_image = resize_image(img_rgb)
                st.image(resized_image, caption='Resized Image', use_column_width=True)

            if function == "Flip Image":
                flipped_image = flip_image(img_rgb)
                st.image(flipped_image, caption='Flipped Image', use_column_width=True)

            # Define convolution matrices for different filters
            identity_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            box_blur_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            gaussian_blur_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
            edge_detection_horizontal_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            edge_detection_vertical_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            emboss_filter = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            outline_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            identity_enhanced_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

            # Map filter names to their respective matrices
            filter_mapping = {
                "Identity Filter": identity_filter,
                "Box Blur": box_blur_filter,
                "Gaussian Blur": gaussian_blur_filter,
                "Edge Detection (Horizontal)": edge_detection_horizontal_filter,
                "Edge Detection (Vertical)": edge_detection_vertical_filter,
                "Emboss": emboss_filter,
                "Sharpen": sharpen_filter,
                "Outline": outline_filter,
                "Identity Enhanced": identity_enhanced_filter
            }


            if function == "Convolution Filter":
                # Select a filter
                selected_filter = st.selectbox("Select a filter", list(filter_mapping.keys()))

                # Use the selected filter matrix
                conv_matrix = filter_mapping[selected_filter]

                # Apply convolution
                conv_result = apply_convolution(img_rgb, conv_matrix)

                st.image(conv_result, caption=f'{selected_filter} Result', use_column_width=True)

            if function == "Detect Faces":
                detected_faces_image, num_faces, faces_rect = detect_faces(img)

                if detected_faces_image is not None:
                    st.image(detected_faces_image, caption=f'Detected Faces ({num_faces} faces)', use_column_width=True)

                    # Load emotion recognition model
                    json_file = open('model/emotion_model.json', 'r')
                    loaded_model_json = json_file.read()
                    json_file.close()
                    emotion_model = model_from_json(loaded_model_json)

                    # load weights into new model
                    emotion_model.load_weights("model/emotion_model.h5")

                    # Classify emotion
                    img_with_emotion, emotions = classify_emotion(detected_faces_image, emotion_model, faces_rect)

                    # Display the result
                    st.image(img_with_emotion, caption=f'Detected Faces with Emotion Labels', use_column_width=True)
                else:
                    st.write("Error detecting faces.")

    elif page == "Table":
        st.markdown("<h2 class='h'>Table Page</h2>", unsafe_allow_html=True)
        st.write("<p class='header'>Upload a CSV file to display as a table.</p>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Display a caption above the table
            st.markdown("### Uploaded Table")

            st.write("The length of this DataFrame is **" + str(len(df)) + "**")

            # Display the uploaded table
            st.write("This is the dataframe")
            st.write(df)

            # Methods
            method = st.selectbox("Select a method",
                                  ["Select a method", "Dataframe and statistics", "Visualization", "Training model"])

            if method == "Select a method":
                st.markdown("<p class='header'>Select a method from the select box </p>", unsafe_allow_html=True)

            if method == "Dataframe and statistics":
                # function
                function = st.selectbox("Select a function",
                                        ["Select a function", "Top 5 rows", "Bottom 5 rows", "Column names",
                                         "Random Sample of 5 rows",
                                         "A specific row", "Specific rows", "Statistics"])

                if function == "Select a function":
                    st.markdown("<p class='header'>Select a function from the select box </p>", unsafe_allow_html=True)

                if function == "Top 5 rows":
                    the_top5_rows = top_5_rows(df)
                    st.write(the_top5_rows)

                if function == "Bottom 5 rows":
                    the_bottom5_rows = bottom_5_rows(df)
                    st.write(the_bottom5_rows)

                if function == "Column names":
                    the_columns_name = columns_name(df)
                    st.write("Column Names:")
                    st.write(the_columns_name)

                if function == "Random Sample of 5 rows":
                    random_rows10 = random_rows(df)
                    st.write(random_rows10)

                if function == "A specific row":
                    # Asking user to write the rank of the row
                    rank = st.number_input("Enter the rank you want to see", min_value=0, max_value=len(df) - 1)
                    specific_row = a_specific_row(df, rank)
                    st.write(specific_row)

                if function == "Specific rows":
                    # Asking user to write the ranks of the row
                    rank1 = st.number_input("Enter the first row", min_value=0, max_value=len(df) - 2)
                    rank2 = st.number_input("Enter the last row", min_value=rank1 + 1, max_value=len(df) - 1)
                    specific_row = specific_rows(df, rank1, rank2)
                    st.write(specific_row)

                if function == "Statistics":

                    the_columns_name = columns_name(df)
                    numerical_dataset = df.select_dtypes(include=['number'])
                    numerical_dataset_names = columns_name(numerical_dataset)
                    categorical_dataset = df.select_dtypes(include=['object'])
                    categorical_dataset_names = columns_name(categorical_dataset)

                    my_columns = []
                    # Use a counter variable to generate unique keys
                    counter = 0

                    st.write(f"The dataframe has **{len(df.columns)}** columns")
                    st.write(the_columns_name)
                    if len(numerical_dataset.columns) == 0:
                        st.write("There is no numerical variable in this dataframe.")

                    elif len(df.columns) == len(numerical_dataset.columns):
                        st.write("All of them are numerical variables")

                        grouping_category = st.selectbox("Group by", the_columns_name)

                        number_of_columns = st.number_input("Enter the number of columns", min_value=1,
                                                             max_value=len(df.columns) - 1)

                        for column in range(number_of_columns):
                            column = st.selectbox(f"Select column {counter + 1}", the_columns_name,
                                                  key=f"column_{counter}")
                            my_columns.append(column)
                            counter += 1

                        # Check for identical columns in the selected choices
                        if len(my_columns) != len(set(my_columns)):
                            st.write("Columns you chose are identical, please choose different columns.")
                        else:

                            action = st.selectbox("Select an action",
                                                  ["Mean", "Sum", "Median", "Variance",
                                                   "Standard Deviation", "Range"], key="action")
                            stats_on_variables = statistics(df, action, grouping_category, my_columns)
                            st.write(stats_on_variables)

                    else:
                       grouping_category = st.selectbox("Group by", categorical_dataset_names)

                       number_of_columns = st.number_input("Enter the number of columns", min_value=1,
                                                            max_value=len(numerical_dataset.columns))

                       for column in range(number_of_columns):
                           column = st.selectbox(f"Select column {counter + 1}", numerical_dataset_names,
                                                 key=f"column_{counter}")
                           my_columns.append(column)
                           counter += 1

                       if len(my_columns) != len(set(my_columns)):
                           st.write("Columns you chose are identical, please choose different columns.")
                       else:
                           action = st.selectbox("Select an action",
                                                 ["Select an action", "Mean", "Sum", "Median", "Variance",
                                                  "Standard Deviation", "Range"], key="action")

                           stats_on_variables = statistics(df, action, grouping_category, my_columns)
                           st.write(stats_on_variables)

            if method == "Visualization":
                # function
                plot = st.selectbox("Select a type of plot", ["Select a type of plot", "Line plot", "Scatter plot",
                                                              "Box plot", "Histogram", "KDE plot", "Violin plot",
                                                              "Bar plot", "Heatmap", "Pie chart", "Pair plot"])

                if plot == "Select a type of plot":
                    st.markdown("<p class='header'>Select a type of plot from the select box </p>",
                                unsafe_allow_html=True)
                else:
                    st.write(f"The dataframe has **{len(df.columns)}** columns")
                    the_columns_name = columns_name(df)
                    st.write(the_columns_name)

                    my_axis = []

                    # Use a counter variable to generate unique keys
                    counter = 0

                    if plot == "Line plot":
                        numerical_dataset = df.select_dtypes(include=['number'])
                        numerical_dataset_names = columns_name(numerical_dataset)

                        for axis in range(2):
                            axis = st.selectbox(f"Select axis {counter + 1}", numerical_dataset_names, key=f"axis_{counter}")
                            my_axis.append(axis)
                            counter += 1
                        line_plot(numerical_dataset, my_axis[0], my_axis[1])

                    if plot == "Scatter plot":
                        numerical_dataset = df.select_dtypes(include=['number'])
                        numerical_dataset_names = columns_name(numerical_dataset)

                        for axis in range(2):
                            axis = st.selectbox(f"Select axis {counter + 1}", numerical_dataset_names, key=f"axis_{counter}")
                            my_axis.append(axis)
                            counter += 1
                        scatter_plot(numerical_dataset, my_axis[0], my_axis[1])

                    if plot == "Box plot":
                        numerical_dataset = df.select_dtypes(include=['number'])
                        categorical_dataset = df.select_dtypes(include=['object'])

                        dataset_numerical_columns_names = columns_name(numerical_dataset)
                        dataset_categorical_columns_names = columns_name(categorical_dataset)

                        # Convert lists to DataFrames
                        numerical_df = pd.DataFrame({'Numerical variable': dataset_numerical_columns_names})
                        categorical_df = pd.DataFrame({'Categorical variable': dataset_categorical_columns_names})

                        # Merge the two DataFrames side by side
                        dataset_merged = pd.concat([numerical_df, categorical_df], axis=1)

                        # Show them side by side with Streamlit
                        st.write(dataset_merged)

                        num_var = st.selectbox(f"Select the numerical variable", dataset_numerical_columns_names)
                        cat_var = st.selectbox(f"Select the categorical variable", dataset_categorical_columns_names)

                        box_plot(df, num_var, cat_var)

                    if plot == "Histogram":
                        numerical_dataset = df.select_dtypes(include=['object'])
                        numerical_dataset_names = columns_name(numerical_dataset)
                        axis = st.selectbox(f"Select axis ", numerical_dataset_names)
                        histogram(numerical_dataset, axis)

                    if plot == "KDE plot":
                        numerical_dataset = df.select_dtypes(include=['number'])
                        numerical_dataset_names = columns_name(numerical_dataset)
                        axis = st.selectbox(f"Select axis ", numerical_dataset_names)
                        kde_plot(numerical_dataset, axis)

                    if plot == "Violin plot":
                        numerical_dataset = df.select_dtypes(include=['number'])
                        categorical_dataset = df.select_dtypes(include=['object'])

                        dataset_numerical_columns_names = columns_name(numerical_dataset)
                        dataset_categorical_columns_names = columns_name(categorical_dataset)

                        # Convert lists to DataFrames
                        numerical_df = pd.DataFrame({'Numerical variable': dataset_numerical_columns_names})
                        categorical_df = pd.DataFrame({'Categorical variable': dataset_categorical_columns_names})

                        # Merge the two DataFrames side by side
                        dataset_merged = pd.concat([numerical_df, categorical_df], axis=1)

                        # Show them side by side with Streamlit
                        st.write(dataset_merged)

                        num_var = st.selectbox(f"Select the numerical variable", dataset_numerical_columns_names)
                        cat_var = st.selectbox(f"Select the categorical variable", dataset_categorical_columns_names)

                        violin_plot(df, num_var, cat_var)

                    if plot == "Bar plot":
                        numerical_dataset = df.select_dtypes(include=['number'])
                        categorical_dataset = df.select_dtypes(include=['object'])

                        dataset_numerical_columns_names = columns_name(numerical_dataset)
                        dataset_categorical_columns_names = columns_name(categorical_dataset)

                        # Convert lists to DataFrames
                        numerical_df = pd.DataFrame({'Numerical variable': dataset_numerical_columns_names})
                        categorical_df = pd.DataFrame({'Categorical variable': dataset_categorical_columns_names})

                        # Merge the two DataFrames side by side
                        dataset_merged = pd.concat([numerical_df, categorical_df], axis=1)

                        # Show them side by side with Streamlit
                        st.write(dataset_merged)

                        num_var = st.selectbox(f"Select the numerical variable", dataset_numerical_columns_names)
                        cat_var = st.selectbox(f"Select the categorical variable", dataset_categorical_columns_names)

                        bar_plot(df, num_var, cat_var)

                    if plot == "Heatmap":
                        dataset = df.select_dtypes(include=['number'])
                        st.write("Columns that the type is **number**")
                        the_dataset_columns = columns_name(dataset)
                        st.write(the_dataset_columns)
                        heat_map(dataset)

                    if plot == "Pie chart":
                        dataset = df.select_dtypes(include=['object'])
                        st.write("Columns that the type is **number**")
                        the_dataset_columns = columns_name(dataset)
                        st.write(the_dataset_columns)
                        axis = st.selectbox(f"Select a column", the_dataset_columns)
                        pie_chart(dataset, axis)

                    if plot == "Pair plot":
                        pair_plot(df)

            if method == "Training model":
                #function
                function = st.selectbox("Select a method", ["Select a method", "Linear regression",
                                                            "Classification with SVM"])

                if function == "Select a method":
                    st.markdown("<p class='header'>Select a method from the select box </p>",
                                unsafe_allow_html=True)

                if function == "Linear regression":
                    numerical_dataset = df.select_dtypes(include=['number'])
                    numerical_dataset_names = columns_name(numerical_dataset)
                    st.write("Available columns:", numerical_dataset_names)

                    st.write("Linear regression is represented by the following equation : **y = ax + b**")

                    y_column_name = st.selectbox("Select your y", numerical_dataset_names)
                    y = numerical_dataset[y_column_name]
                    x = numerical_dataset.drop([y_column_name], axis=1)
                    st.write(y)
                    st.write("These are your x :")
                    st.write(x)

                    linear_regression(x, y, x)

                if function == "Classification with SVM":
                    the_columns_name = columns_name(df)
                    st.write("Available columns:", the_columns_name)

                    categorical_dataset = df.select_dtypes(include=['object'])
                    numerical_dataset = df.select_dtypes(include=['number'])

                    categorical_dataset_names = columns_name(categorical_dataset)
                    numerical_dataset_names = columns_name(numerical_dataset)

                    y_column_name = st.selectbox("Select your y", categorical_dataset_names)
                    if y_column_name in categorical_dataset.columns:
                        y = categorical_dataset[y_column_name]
                        st.write(y)

                        st.write("These are your x:")
                        x = numerical_dataset
                        classification_with_SVM(y, x)
                    else:
                        st.write("Invalid column name or None provided for y_column_name.")

    elif page == "Text":
        st.markdown("<h2 class='h'>Text Page</h2>", unsafe_allow_html=True)
        st.write("<p class='header'>Upload a text file for processing.</p>", unsafe_allow_html=True)
        uploaded_text = st.file_uploader("Choose a text file", type=["txt"])
        if uploaded_text is not None:
            # Process the uploaded text
            text_content = uploaded_text.read().decode("utf-8")
            # Perform text processing here
            st.write("Here is what is written in your file:")
            st.write(text_content)

            option = st.selectbox("Select an option", ["Select an option", "Extract emails", "Length of the text",
                                                       "Verify color", "Detect language", "Clean text",
                                                       "Remove stopwords", "Analyze sentiment"])

            if option == "Select an option":
                st.markdown("<p class='header'>Select an option from the select box </p>",
                            unsafe_allow_html=True)

            if option == "Extract emails":
                st.write(extract_email(text_content))

            if option == "Length of the text":
                st.write(text_lenght(text_content))

            if option == "Verify color":
                st.write(verify_color(text_content))

            if option == "Detect language":
                st.write(detect_language(text_content))

            if option == "Clean text":
                cleaned_text = clean_text(text_content)
                st.write("Cleaned text:")
                st.write(cleaned_text)

            if option == "Remove stopwords":
                text_without_stopwords = remove_stopwords(text_content)
                st.write("Text without stopwords:")
                st.write(text_without_stopwords)

            if option == "Analyze sentiment":
                sentiment = analyze_sentiment(text_content)
                st.write(f"Sentiment: {sentiment}")

    elif page == "CNN image classifier":
        st.markdown("<h3 class='h'>Dog vs cat Image Classifier with CNN : </h3> <br/>", unsafe_allow_html=True)
        X_train, Y_train, X_test, Y_test = load_dataset()
        display_random_image(X_train, image_caption="Random Training Image", width=300)
        model = create_model()
        train_model(model, X_train, Y_train, epochs=5, batch_size=64)
        evaluation_result = evaluate_model(model, X_test, Y_test)
        st.write(f'Model evaluation result: {evaluation_result}')
        display_random_test_image(X_test, Y_test, model, image_caption="Random Test Image", width=300)


if __name__ == "__main__":
    main()