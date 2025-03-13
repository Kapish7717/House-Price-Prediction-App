<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# App title and description
st.title("House Price Prediction App")
st.write("This app predicts house prices using linear regression based on features like size, bedrooms, etc.")

# Step 1: Data loading
@st.cache_data
def load_sample_data():
    # For demonstration, generate some sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic house data
    size = np.random.randint(500, 5000, n_samples)  # house size in sq ft
    bedrooms = np.random.randint(1, 7, n_samples)   # number of bedrooms
    bathrooms = np.random.randint(1, 5, n_samples)  # number of bathrooms
    age = np.random.randint(0, 50, n_samples)       # age of house in years
    
    # Generate price with some noise
    price = 50000 + 100 * size + 25000 * bedrooms + 35000 * bathrooms - 1000 * age
    price = price + np.random.normal(0, 50000, n_samples)  # add some noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'size': size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'price': price
    })
    
    return data

# Data source selection
st.header("Data Source")
data_source = st.radio(
    "Choose your data source:",
    ["Upload your own CSV file", "Use sample data"]
)

# Initialize data variable
data = None

# Handle file upload
if data_source == "Upload your own CSV file":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Allow user to select the target column
            target_column = st.selectbox(
                "Select the column that contains the house prices (target variable):",
                options=data.columns.tolist()
            )
            
            # Set the target column as 'price' for consistency in the app
            data = data.rename(columns={target_column: 'price'})
            
            # Display data types and allow user to select numerical columns
            st.write("Column data types:")
            st.write(data.dtypes)
            
            # Select features (only numerical columns except the target)
            numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if 'price' in numerical_columns:
                numerical_columns.remove('price')
            
            feature_columns = st.multiselect(
                "Select the feature columns to use for prediction:",
                options=numerical_columns,
                default=numerical_columns
            )
            
            # Filter data to include only selected features and target
            if feature_columns:
                data = data[feature_columns + ['price']]
            else:
                st.error("Please select at least one feature column.")
                data = None
        except Exception as e:
            st.error(f"Error: {e}")
            st.error("Please ensure your CSV file is properly formatted with numerical data.")
            data = None
else:
    # Use sample data
    data = load_sample_data()
    st.info("Using sample data with features: size, bedrooms, bathrooms, age and price.")

# Continue only if data is loaded
if data is not None:
    # Step 2: Data overview section
    st.header("Data Overview")
    if st.checkbox("Show raw data"):
        st.write(data.head(10))

    st.write(f"Dataset contains {data.shape[0]} houses with {data.shape[1]} features.")

    # Show basic statistics
    if st.checkbox("Show statistics"):
        st.write(data.describe())
        
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        st.warning("Your dataset contains missing values:")
        st.write(missing_values[missing_values > 0])
        
        # Handle missing values
        if st.button("Remove rows with missing values"):
            data = data.dropna()
            st.success(f"Removed rows with missing values. New dataset shape: {data.shape}")

    # Step 3: Data visualization
    st.header("Data Visualization")
    if len(data.columns) > 1:  # Make sure there's at least one feature besides price
        feature_to_plot = st.selectbox("Select feature to visualize against price", 
                                      [col for col in data.columns if col != 'price'])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data[feature_to_plot], data['price'], alpha=0.5)
        ax.set_xlabel(feature_to_plot)
        ax.set_ylabel('Price')
        ax.set_title(f"{feature_to_plot.capitalize()} vs. Price")
        st.pyplot(fig)

    # Step 4: Model building section
    st.header("Build and Train Model")

    # Let user select features
    st.subheader("Select Features for Training")
    available_features = [col for col in data.columns if col != 'price']
    selected_features = st.multiselect(
        "Choose features to include in the model",
        options=available_features,
        default=available_features
    )

    if not selected_features:
        st.error("Please select at least one feature.")
    else:
        # Prepare data
        X = data[selected_features]
        y = data['price']
        
        # Split data
        test_size = st.slider("Select test data percentage", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Display metrics
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Training Set:")
                    st.write(f"RMSE: ${train_rmse:.2f}")
                    st.write(f"R²: {train_r2:.4f}")
                with col2:
                    st.write("Test Set:")
                    st.write(f"RMSE: ${test_rmse:.2f}")
                    st.write(f"R²: {test_r2:.4f}")
                
                # Show coefficients
                st.subheader("Model Coefficients")
                coef_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Coefficient': model.coef_
                })
                st.write(f"Intercept: ${model.intercept_:.2f}")
                st.write(coef_df)
                
                # Plot actual vs predicted prices
                st.subheader("Actual vs Predicted Prices")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, y_pred_test, alpha=0.5)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                ax.set_xlabel("Actual Price")
                ax.set_ylabel("Predicted Price")
                ax.set_title("Actual vs Predicted House Prices")
                st.pyplot(fig)
                
                # Save model for prediction
                st.session_state['model'] = model
                st.session_state['features'] = selected_features
                st.success("Model trained successfully! You can now make predictions.")

        # Step 5: Make predictions
        st.header("Make Predictions")
        st.write("Enter house details to predict its price")

        # Initialize input columns
        col1, col2 = st.columns(2)

        # Create input fields based on selected features
        user_input = {}
        if 'features' in st.session_state:
            # Split features across two columns
            features_left = st.session_state['features'][:len(st.session_state['features'])//2 + len(st.session_state['features'])%2]
            features_right = st.session_state['features'][len(st.session_state['features'])//2 + len(st.session_state['features'])%2:]
            
            # Left column inputs
            for feature in features_left:
                min_val = float(data[feature].min())
                max_val = float(data[feature].max())
                default_val = float((min_val + max_val) / 2)
                user_input[feature] = col1.number_input(
                    f"{feature.capitalize()}", 
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val
                )
            
            # Right column inputs
            for feature in features_right:
                min_val = float(data[feature].min())
                max_val = float(data[feature].max())
                default_val = float((min_val + max_val) / 2)
                user_input[feature] = col2.number_input(
                    f"{feature.capitalize()}", 
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val
                )
            
            # Make prediction when button is clicked
            if st.button("Predict Price"):
                if 'model' in st.session_state:
                    # Create input DataFrame
                    input_df = pd.DataFrame([user_input])
                    
                    # Make prediction
                    prediction = st.session_state['model'].predict(input_df)[0]
                    
                    # Display prediction
                    st.success(f"Predicted House Price: ${prediction:,.2f}")
                    
                    # Show explanation of prediction
                    st.subheader("Explanation")
                    st.write("This prediction is based on the following:")
                    
                    explanation = pd.DataFrame({
                        'Feature': list(user_input.keys()),
                        'Value': list(user_input.values()),
                        'Coefficient': st.session_state['model'].coef_,
                        'Contribution': np.array(list(user_input.values())) * st.session_state['model'].coef_
                    })
                    explanation['Contribution'] = explanation['Contribution'].round(2)
                    st.write(explanation)
                    
                    # Add base price (intercept)
                    st.write(f"Base price (intercept): ${st.session_state['model'].intercept_:.2f}")
                    st.write(f"Sum of all contributions: ${explanation['Contribution'].sum() + st.session_state['model'].intercept_:.2f}")
                else:
                    st.error("Please train the model first!")
        else:
            st.info("Please select features and train the model before making predictions.")

    # Step 6: Download trained model
    if 'model' in st.session_state:
        st.header("Download Model Results")
        
        # Create a download button for model coefficients
        coef_df = pd.DataFrame({
            'Feature': st.session_state['features'],
            'Coefficient': st.session_state['model'].coef_
        })
        
        # Add intercept
        intercept_df = pd.DataFrame({
            'Feature': ['Intercept'],
            'Coefficient': [st.session_state['model'].intercept_]
        })
        
        # Combine
        model_results = pd.concat([intercept_df, coef_df], ignore_index=True)
        
        # Create CSV in memory
        csv = model_results.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="Download Model Coefficients",
            data=csv,
            file_name="house_price_model_coefficients.csv",
            mime="text/csv",
=======
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="House Price Prediction App", layout="wide")

st.title("House Price Prediction App")
st.write("This app predicts house prices using linear regression based on features like size, bedrooms, etc.")

@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    size = np.random.randint(500, 5000, n_samples) 
    bedrooms = np.random.randint(1, 7, n_samples)  
    bathrooms = np.random.randint(1, 5, n_samples) 
    age = np.random.randint(0, 50, n_samples)       
    
    price = 50000 + 100 * size + 25000 * bedrooms + 35000 * bathrooms - 1000 * age
    price = price + np.random.normal(0, 50000, n_samples)  # add some noise
    
    data = pd.DataFrame({
        'size': size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'price': price
    })
    
    return data

st.header("Data Source")
data_source = st.radio(
    "Choose your data source:",
    ["Upload your own CSV file", "Use sample data"]
)

data = None

if data_source == "Upload your own CSV file":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Allow user to select the target column
            target_column = st.selectbox(
                "Select the column that contains the house prices (target variable):",
                options=data.columns.tolist()
            )
            
            data = data.rename(columns={target_column: 'price'})
            
            st.write("Column data types:")
            st.write(data.dtypes)
            
            numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if 'price' in numerical_columns:
                numerical_columns.remove('price')
            
            feature_columns = st.multiselect(
                "Select the feature columns to use for prediction:",
                options=numerical_columns,
                default=numerical_columns
            )
            
            if feature_columns:
                data = data[feature_columns + ['price']]
            else:
                st.error("Please select at least one feature column.")
                data = None
        except Exception as e:
            st.error(f"Error: {e}")
            st.error("Please ensure your CSV file is properly formatted with numerical data.")
            data = None
else:
    data = load_sample_data()
    st.info("Using sample data with features: size, bedrooms, bathrooms, age and price.")

if data is not None:
    st.header("Data Overview")
    if st.checkbox("Show raw data"):
        st.write(data.head(10))

    st.write(f"Dataset contains {data.shape[0]} houses with {data.shape[1]} features.")

    if st.checkbox("Show statistics"):
        st.write(data.describe())
        
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        st.warning("Your dataset contains missing values:")
        st.write(missing_values[missing_values > 0])
        
        if st.button("Remove rows with missing values"):
            data = data.dropna()
            st.success(f"Removed rows with missing values. New dataset shape: {data.shape}")

    st.header("Data Visualization")
    if len(data.columns) > 1:  
        feature_to_plot = st.selectbox("Select feature to visualize against price", 
                                      [col for col in data.columns if col != 'price'])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data[feature_to_plot], data['price'], alpha=0.5)
        ax.set_xlabel(feature_to_plot)
        ax.set_ylabel('Price')
        ax.set_title(f"{feature_to_plot.capitalize()} vs. Price")
        st.pyplot(fig)

    st.header("Build and Train Model")

    st.subheader("Select Features for Training")
    available_features = [col for col in data.columns if col != 'price']
    selected_features = st.multiselect(
        "Choose features to include in the model",
        options=available_features,
        default=available_features
    )

    if not selected_features:
        st.error("Please select at least one feature.")
    else:
        X = data[selected_features]
        y = data['price']
        
        test_size = st.slider("Select test data percentage", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Training Set:")
                    st.write(f"RMSE: ${train_rmse:.2f}")
                    st.write(f"R²: {train_r2:.4f}")
                with col2:
                    st.write("Test Set:")
                    st.write(f"RMSE: ${test_rmse:.2f}")
                    st.write(f"R²: {test_r2:.4f}")
                
                st.subheader("Model Coefficients")
                coef_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Coefficient': model.coef_
                })
                st.write(f"Intercept: ${model.intercept_:.2f}")
                st.write(coef_df)
                
                st.subheader("Actual vs Predicted Prices")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, y_pred_test, alpha=0.5)
                ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                ax.set_xlabel("Actual Price")
                ax.set_ylabel("Predicted Price")
                ax.set_title("Actual vs Predicted House Prices")
                st.pyplot(fig)
                
                st.session_state['model'] = model
                st.session_state['features'] = selected_features
                st.success("Model trained successfully! You can now make predictions.")

        st.header("Make Predictions")
        st.write("Enter house details to predict its price")

        col1, col2 = st.columns(2)

        user_input = {}
        if 'features' in st.session_state:
            features_left = st.session_state['features'][:len(st.session_state['features'])//2 + len(st.session_state['features'])%2]
            features_right = st.session_state['features'][len(st.session_state['features'])//2 + len(st.session_state['features'])%2:]
            
            for feature in features_left:
                min_val = float(data[feature].min())
                max_val = float(data[feature].max())
                default_val = float((min_val + max_val) / 2)
                user_input[feature] = col1.number_input(
                    f"{feature.capitalize()}", 
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val
                )
            
            for feature in features_right:
                min_val = float(data[feature].min())
                max_val = float(data[feature].max())
                default_val = float((min_val + max_val) / 2)
                user_input[feature] = col2.number_input(
                    f"{feature.capitalize()}", 
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val
                )
            
            if st.button("Predict Price"):
                if 'model' in st.session_state:
                    input_df = pd.DataFrame([user_input])
                    
                    prediction = st.session_state['model'].predict(input_df)[0]
                    
                    st.success(f"Predicted House Price: ${prediction:,.2f}")
                    
                    st.subheader("Explanation")
                    st.write("This prediction is based on the following:")
                    
                    explanation = pd.DataFrame({
                        'Feature': list(user_input.keys()),
                        'Value': list(user_input.values()),
                        'Coefficient': st.session_state['model'].coef_,
                        'Contribution': np.array(list(user_input.values())) * st.session_state['model'].coef_
                    })
                    explanation['Contribution'] = explanation['Contribution'].round(2)
                    st.write(explanation)
                    
                    st.write(f"Base price (intercept): ${st.session_state['model'].intercept_:.2f}")
                    st.write(f"Sum of all contributions: ${explanation['Contribution'].sum() + st.session_state['model'].intercept_:.2f}")
                else:
                    st.error("Please train the model first!")
        else:
            st.info("Please select features and train the model before making predictions.")

    if 'model' in st.session_state:
        st.header("Download Model Results")
        
        coef_df = pd.DataFrame({
            'Feature': st.session_state['features'],
            'Coefficient': st.session_state['model'].coef_
        })
        
        intercept_df = pd.DataFrame({
            'Feature': ['Intercept'],
            'Coefficient': [st.session_state['model'].intercept_]
        })
        
        model_results = pd.concat([intercept_df, coef_df], ignore_index=True)
        
        csv = model_results.to_csv(index=False)
        
        st.download_button(
            label="Download Model Coefficients",
            data=csv,
            file_name="house_price_model_coefficients.csv",
            mime="text/csv",
>>>>>>> 811bef1 (Initial commit)
        )