import gradio as gr
import pandas as pd
import numpy as np
import joblib


# =========================================================
# 1. LOAD SAVED MODEL / PREPROCESSING FILES
# =========================================================

MODEL_PATH = "best_xgboost_churn_model.pkl"
IMPUTER_PATH = "churn_imputer.pkl"
FEATURE_PATH = "churn_feature_names.pkl"
THRESHOLD_PATH = "churn_thresholds.pkl"


model = joblib.load(MODEL_PATH)

imputer = joblib.load(IMPUTER_PATH)

feature_names = joblib.load(FEATURE_PATH)

thresholds = joblib.load(THRESHOLD_PATH)


support_threshold = thresholds["support_threshold"]

complaint_threshold = thresholds["complaint_threshold"]


print("=" * 60)
print("MODEL LOADED SUCCESSFULLY")
print("Number of model features:", len(feature_names))
print("Support threshold:", support_threshold)
print("Complaint threshold:", complaint_threshold)
print("=" * 60)


# =========================================================
# 2. PREDICTION FUNCTION
# =========================================================

def predict_churn(

    # -----------------------------
    # CUSTOMER DETAILS
    # -----------------------------

    age,
    gender,
    city,
    income_level,
    tenure_months,
    signup_date,
    contract_type,
    plan_type,
    payment_method,
    auto_renewal,

    # -----------------------------
    # CUSTOMER BEHAVIOUR
    # -----------------------------

    usage_frequency,
    days_since_last_activity,
    feature_usage_count,
    avg_session_duration,
    monthly_charges,
    total_charges,
    discount_applied,
    late_payment_count,
    support_tickets,
    complaint_count,
    avg_resolution_time,
    satisfaction_score,
    referral_count,
    loyalty_program,
    upsell_downgrade,
    email_open_rate
):

    try:

        # =================================================
        # 3. CREATE RAW DATAFRAME
        # =================================================

        data = pd.DataFrame({

            "Customer_ID": ["NEW_CUSTOMER"],

            "Age": [age],

            "Gender": [gender],

            "City": [city],

            "Income_Level": [income_level],

            "Tenure_Months": [tenure_months],

            "Signup_Date": [signup_date],

            "Contract_Type": [contract_type],

            "Plan_Type": [plan_type],

            "Payment_Method": [payment_method],

            "Auto_Renewal_Flag": [auto_renewal],

            "Usage_Frequency": [
                usage_frequency
            ],

            "Days_Since_Last_Activity": [
                days_since_last_activity
            ],

            "Feature_Usage_Count": [
                feature_usage_count
            ],

            "Avg_Session_Duration_Min": [
                avg_session_duration
            ],

            "Monthly_Charges": [
                monthly_charges
            ],

            "Total_Charges": [
                total_charges
            ],

            "Discount_Applied_Pct": [
                discount_applied
            ],

            "Late_Payment_Count": [
                late_payment_count
            ],

            "Support_Tickets_Raised": [
                support_tickets
            ],

            "Complaint_Count": [
                complaint_count
            ],

            "Avg_Resolution_Time_Hrs": [
                avg_resolution_time
            ],

            "Satisfaction_Score": [
                satisfaction_score
            ],

            "Referral_Count": [
                referral_count
            ],

            "Loyalty_Program_Member": [
                loyalty_program
            ],

            "Upsell_Downgrade_History": [
                upsell_downgrade
            ],

            "Email_Open_Rate_Pct": [
                email_open_rate
            ]
        })


        # =================================================
        # 4. CONVERT DATE
        # =================================================

        data["Signup_Date"] = pd.to_datetime(
            data["Signup_Date"],
            errors="coerce"
        )

        if data["Signup_Date"].isna().any():

            return (
                "❌ Invalid Signup Date",
                "0.00%",
                "0.00%"
            )


        # =================================================
        # 5. DATE FEATURE ENGINEERING
        # =================================================

        data["Signup_Year"] = (
            data["Signup_Date"].dt.year
        )

        data["Signup_Month"] = (
            data["Signup_Date"].dt.month
        )

        data["Signup_Day"] = (
            data["Signup_Date"].dt.day
        )

        data["Signup_DayOfWeek"] = (
            data["Signup_Date"].dt.dayofweek
        )

        data["Signup_Is_Weekend"] = (
            data["Signup_DayOfWeek"] >= 5
        ).astype(int)


        # =================================================
        # 6. TENURE YEARS
        # =================================================

        data["Tenure_Years"] = (
            data["Tenure_Months"] / 12
        )


        # =================================================
        # 7. TOTAL USAGE TIME
        # =================================================

        data["Total_Usage_Time"] = (
            data["Usage_Frequency"]
            *
            data["Avg_Session_Duration_Min"]
        )


        # =================================================
        # 8. TOTAL SUPPORT ISSUES
        # =================================================

        data["Total_Support_Issues"] = (
            data["Support_Tickets_Raised"]
            +
            data["Complaint_Count"]
        )


        # =================================================
        # 9. HAS LATE PAYMENT
        # =================================================

        data["Has_Late_Payment"] = (
            data["Late_Payment_Count"] > 0
        ).astype(int)


        # =================================================
        # 10. HAS REFERRAL
        # =================================================

        data["Has_Referral"] = (
            data["Referral_Count"] > 0
        ).astype(int)


        # =================================================
        # 11. AVERAGE MONTHLY CHARGE
        # =================================================

        data["Avg_Monthly_Charge"] = (
            data["Total_Charges"]
            /
            data["Tenure_Months"].replace(
                0,
                np.nan
            )
        )


        # =================================================
        # 12. HIGH SUPPORT USAGE
        # =================================================

        data["High_Support_Usage"] = (
            data["Support_Tickets_Raised"]
            >
            support_threshold
        ).astype(int)


        # =================================================
        # 13. HIGH COMPLAINT
        # =================================================

        data["High_Complaint"] = (
            data["Complaint_Count"]
            >
            complaint_threshold
        ).astype(int)


        # =================================================
        # 14. ENGAGEMENT LEVEL
        # =================================================

        data["Engagement_Level"] = pd.cut(

            data["Usage_Frequency"],

            bins=[
                -np.inf,
                10,
                30,
                np.inf
            ],

            labels=[
                "Low",
                "Medium",
                "High"
            ]
        )


        # =================================================
        # 15. REMOVE CUSTOMER ID
        # =================================================

        data = data.drop(
            columns=["Customer_ID"],
            errors="ignore"
        )


        # =================================================
        # 16. GENDER ENCODING
        # =================================================

        data["Gender"] = data["Gender"].map({

            "Male": 0,
            "Female": 1,
            "Other": 2

        })


        # =================================================
        # 17. YES / NO ENCODING
        # =================================================

        binary_columns = [

            col

            for col in data.columns

            if set(
                data[col].dropna().unique()
            ) <= {"Yes", "No"}

        ]


        for col in binary_columns:

            data[col] = data[col].map({

                "No": 0,
                "Yes": 1

            })


        # =================================================
        # 18. ENGAGEMENT ENCODING
        # =================================================

        data["Engagement_Level"] = (
            data["Engagement_Level"].map({

                "Low": 0,
                "Medium": 1,
                "High": 2

            })
        )


        # =================================================
        # 19. REMOVE SIGNUP DATE
        # =================================================

        data = data.drop(
            columns=["Signup_Date"],
            errors="ignore"
        )


        # =================================================
        # 20. ONE-HOT ENCODING
        # =================================================

        categorical_columns = (
            data.select_dtypes(
                include="object"
            ).columns
        )


        data = pd.get_dummies(

            data,

            columns=categorical_columns,

            drop_first=True,

            dtype=int

        )


        # =================================================
        # 21. CHECK FEATURES BEFORE ALIGNMENT
        # =================================================

        missing_features = [
            col
            for col in feature_names
            if col not in data.columns
        ]

        extra_features = [
            col
            for col in data.columns
            if col not in feature_names
        ]


        print("\n" + "=" * 60)
        print("FEATURE DEBUG")
        print("=" * 60)

        print(
            "Expected model features:",
            len(feature_names)
        )

        print(
            "Current input features:",
            len(data.columns)
        )

        print(
            "\nMissing features:",
            missing_features
        )

        print(
            "\nExtra features:",
            extra_features
        )


        # =================================================
        # 22. ALIGN EXACTLY WITH TRAINING FEATURES
        # =================================================

        data = data.reindex(
            columns=feature_names,
            fill_value=0
        )


        print(
            "\nFinal aligned shape:",
            data.shape
        )


        # =================================================
        # 23. CHECK FOR NON-NUMERIC DATA
        # =================================================

        non_numeric_columns = (
            data.select_dtypes(
                exclude=np.number
            ).columns.tolist()
        )

        print(
            "Non-numeric columns:",
            non_numeric_columns
        )


        # =================================================
        # 24. IMPUTATION
        # =================================================

        data_imputed = imputer.transform(
            data
        )


        # =================================================
        # 25. PREDICT PROBABILITY
        # =================================================

        probability = (
            model.predict_proba(
                data_imputed
            )[0][1]
        )

        no_churn_probability = (
            1 - probability
        )


        # =================================================
        # 26. MODEL PREDICTION
        # =================================================

        prediction = model.predict(
            data_imputed
        )[0]


        # =================================================
        # 27. DEBUG PROBABILITY
        # =================================================

        print("\n" + "=" * 60)
        print("MODEL RESULT")
        print("=" * 60)

        print(
            "Churn Probability:",
            probability
        )

        print(
            "No Churn Probability:",
            no_churn_probability
        )

        print(
            "Prediction:",
            prediction
        )

        print("=" * 60)


        # =================================================
        # 28. FINAL RESULT
        # =================================================

        if prediction == 1:

            prediction_text = "⚠️ CHURN"

        else:

            prediction_text = "✅ NO CHURN"


        return (

            prediction_text,

            f"{probability * 100:.2f}%",

            f"{no_churn_probability * 100:.2f}%"

        )


    except Exception as e:

        print("\nERROR:")
        print(str(e))

        return (
            f"❌ Error: {str(e)}",
            "0.00%",
            "0.00%"
        )


# =========================================================
# 29. GRADIO UI
# =========================================================

with gr.Blocks(
    title="Customer Churn Prediction"
) as demo:


    # =====================================================
    # TITLE
    # =====================================================

    gr.Markdown(
        """
        # 📊 Customer Churn Prediction

        Enter customer information to predict
        whether the customer is likely to churn.
        """
    )


    # =====================================================
    # CUSTOMER DETAILS
    # =====================================================

    gr.Markdown(
        """
        ## 👤 Customer Details
        """
    )


    with gr.Row():

        with gr.Column():

            age = gr.Number(
                label="Age",
                value=30,
                minimum=1,
                maximum=100
            )


            gender = gr.Dropdown(

                choices=[
                    "Male",
                    "Female",
                    "Other"
                ],

                value="Male",

                label="Gender"

            )


            city = gr.Dropdown(

                choices=[
                    "Delhi",
                    "Pune",
                    "Chennai",
                    "Kolkata",
                    "Bengaluru",
                    "Mumbai",
                    "Hyderabad"
                ],

                value="Chennai",

                label="City"

            )


            income_level = gr.Dropdown(

                choices=[
                    "Low",
                    "Medium",
                    "High"
                ],

                value="Medium",

                label="Income Level"

            )


            tenure_months = gr.Number(

                label="Tenure Months",

                value=12,

                minimum=0

            )


            signup_date = gr.Textbox(

                label="Signup Date",

                value="2025-01-01",

                placeholder="YYYY-MM-DD"

            )


        with gr.Column():

            contract_type = gr.Dropdown(

                choices=[
                    "Monthly",
                    "Annual",
                    "Prepaid"
                ],

                value="Monthly",

                label="Contract Type"

            )


            plan_type = gr.Dropdown(

                choices=[
                    "Basic",
                    "Standard",
                    "Premium"
                ],

                value="Basic",

                label="Plan Type"

            )


            payment_method = gr.Dropdown(

                choices=[
                    "UPI",
                    "Debit Card",
                    "Wallet",
                    "Net Banking",
                    "Credit Card"
                ],

                value="UPI",

                label="Payment Method"

            )


            auto_renewal = gr.Dropdown(

                choices=[
                    0,
                    1
                ],

                value=1,

                label="Auto Renewal"

            )


            loyalty_program = gr.Dropdown(

                choices=[
                    0,
                    1
                ],

                value=1,

                label="Loyalty Program Member"

            )


            upsell_downgrade = gr.Dropdown(

                choices=[
                    "No History",
                    "Upgraded",
                    "Downgraded"
                ],

                value="No History",

                label="Upsell / Downgrade History"

            )


    # =====================================================
    # CUSTOMER BEHAVIOUR
    # =====================================================

    gr.Markdown(
        """
        ## 📈 Customer Behaviour
        """
    )


    with gr.Row():

        with gr.Column():

            usage_frequency = gr.Number(

                label="Usage Frequency",

                value=20,

                minimum=0

            )


            days_since_last_activity = gr.Number(

                label="Days Since Last Activity",

                value=10,

                minimum=0

            )


            feature_usage_count = gr.Number(

                label="Feature Usage Count",

                value=5,

                minimum=0

            )


            avg_session_duration = gr.Number(

                label="Average Session Duration (Min)",

                value=10,

                minimum=0

            )


            monthly_charges = gr.Number(

                label="Monthly Charges",

                value=500,

                minimum=0

            )


            total_charges = gr.Number(

                label="Total Charges",

                value=6000,

                minimum=0

            )


            discount_applied = gr.Number(

                label="Discount Applied (%)",

                value=0,

                minimum=0,

                maximum=100

            )


            email_open_rate = gr.Number(

                label="Email Open Rate (%)",

                value=50,

                minimum=0,

                maximum=100

            )


        with gr.Column():

            late_payment_count = gr.Number(

                label="Late Payment Count",

                value=0,

                minimum=0

            )


            support_tickets = gr.Number(

                label="Support Tickets Raised",

                value=0,

                minimum=0

            )


            complaint_count = gr.Number(

                label="Complaint Count",

                value=0,

                minimum=0

            )


            avg_resolution_time = gr.Number(

                label="Average Resolution Time (Hours)",

                value=10,

                minimum=0

            )


            satisfaction_score = gr.Slider(

                minimum=1,

                maximum=10,

                value=5,

                step=1,

                label="Satisfaction Score"

            )


            referral_count = gr.Number(

                label="Referral Count",

                value=0,

                minimum=0

            )


    # =====================================================
    # PREDICTION SECTION
    # =====================================================

    gr.Markdown(
        """
        ## 🔮 Prediction
        """
    )


    predict_button = gr.Button(

        "🔮 Predict Churn",

        variant="primary",

        size="lg"

    )


    # =====================================================
    # OUTPUT
    # =====================================================

    with gr.Row():

        prediction_output = gr.Textbox(

            label="Prediction"

        )


        churn_probability_output = gr.Textbox(

            label="Churn Probability"

        )


        no_churn_probability_output = gr.Textbox(

            label="No Churn Probability"

        )


    # =====================================================
    # BUTTON CONNECTION
    # =====================================================

    predict_button.click(

        fn=predict_churn,

        inputs=[

            # Customer Details

            age,

            gender,

            city,

            income_level,

            tenure_months,

            signup_date,

            contract_type,

            plan_type,

            payment_method,

            auto_renewal,


            # Customer Behaviour

            usage_frequency,

            days_since_last_activity,

            feature_usage_count,

            avg_session_duration,

            monthly_charges,

            total_charges,

            discount_applied,

            late_payment_count,

            support_tickets,

            complaint_count,

            avg_resolution_time,

            satisfaction_score,

            referral_count,

            loyalty_program,

            upsell_downgrade,

            email_open_rate

        ],

        outputs=[

            prediction_output,

            churn_probability_output,

            no_churn_probability_output

        ]

    )


# =========================================================
# 30. LAUNCH
# =========================================================

if __name__ == "__main__":

    demo.launch(
        inbrowser=True
    )
    import os
 
    port = int(os.environ.get("PORT", 10000))
 
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )