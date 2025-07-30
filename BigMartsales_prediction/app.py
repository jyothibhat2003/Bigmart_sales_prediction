import streamlit as st
from preprocess import preprocess_input
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🛒 BigMart Sales Prediction ")
st.write("Enter product and outlet details to predict sales.")

# Input fields (make sure categories exactly match training data)
input_dict = {
    "Item_Weight": st.number_input("Item Weight (in kg)", 0.1, 20.0, 5.0),
    "Item_Visibility": st.number_input("Item Visibility", 0.0, 1.0, 0.05),
    "Item_MRP": st.number_input("Item MRP (Maximum Retail Price)", 1.0, 3000.0, 100.0),
    "Outlet_Establishment_Year": st.number_input("Outlet Establishment Year", 1900, 2023, 2005),
    "Item_Fat_Content": st.selectbox("Item Fat Content", ['Low Fat', 'Regular', 'Non-Edible']),
    "Item_Type": st.selectbox("Item Type", [
        'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
        'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
        'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others'
    ]),
    "Outlet_Size": st.selectbox("Outlet Size", ['Small', 'Medium', 'High']),
    "Outlet_Location_Type": st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3']),
    "Outlet_Type": st.selectbox("Outlet Type", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
}

if st.button("Predict Sale"):
    processed = preprocess_input(input_dict)
    prediction = model.predict(processed)[0]
    st.success(f"Predicted Sales: ₹{prediction:.2f}")
