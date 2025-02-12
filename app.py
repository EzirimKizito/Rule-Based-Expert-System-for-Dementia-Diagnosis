import streamlit as st
import pandas as pd

# ---------------------------
# Function to Compute Derived (Interaction) Features
# ---------------------------
def compute_interaction_features(input_data):
    """
    Compute the interaction features from the base inputs.
    
    Expected keys in input_data:
        'MMSE', 'Age', 'ASF', 'nWBV', 'M/F', 'EDUC', 'SES', 'eTIV'
    Returns a dictionary including base features and the derived ones.
    """
    data = {}
    # Base features
    data['MMSE'] = input_data['MMSE']
    data['Age'] = input_data['Age']
    data['ASF'] = input_data['ASF']
    data['nWBV'] = input_data['nWBV']
    data['M/F'] = input_data['M/F']
    data['EDUC'] = input_data['EDUC']
    data['SES'] = input_data['SES']
    data['eTIV'] = input_data['eTIV']

    # Derived interaction features (mined from literature)
    data['Age_MMSE'] = data['Age'] * data['MMSE']                  # Age adjusted by MMSE
    data['Age_nWBV'] = data['Age'] * data['nWBV']                    # Brain volume loss with age
    data['EDUC_MMSE'] = data['EDUC'] * data['MMSE']                  # Education's protective effect
    data['EDUC_nWBV'] = data['EDUC'] * data['nWBV']                    # Education and brain volume relationship
    data['SES_MMSE'] = data['SES'] * data['MMSE']                    # SES impact on cognition
    data['SES_Age'] = data['SES'] * data['Age']                      # SES-adjusted age effect
    data['nWBV_MMSE'] = data['nWBV'] * data['MMSE']                  # Brain structure-cognition coupling
    data['eTIV_ASF'] = data['eTIV'] * data['ASF']                    # Intracranial volume normalization
    data['Gender_Age'] = data['M/F'] * data['Age']                   # Gender-specific aging pattern
    data['Gender_MMSE'] = data['M/F'] * data['MMSE']                 # Gender differences in cognitive scores
    
    # Novel clinically insightful interactions
    if data['nWBV'] != 0 and data['MMSE'] != 0:
        data['Neuro_Risk'] = (data['Age'] / 100) * (1 / data['nWBV']) * (1 / data['MMSE'])
    else:
        data['Neuro_Risk'] = 0
    data['SES_Brain'] = data['SES'] * (data['nWBV'] + data['ASF'])
    if data['MMSE'] != 0 and data['nWBV'] != 0 and data['EDUC'] != 0:
        data['Dementia_Risk_Index'] = (data['Age'] * data['eTIV']) / (data['MMSE'] * data['nWBV'] * data['EDUC'])
    else:
        data['Dementia_Risk_Index'] = 0

    return data

# ---------------------------
# Expert System Definition
# ---------------------------
class ExpertSystem_Group_InteractionFeatures_Classification:
    """
    Expert system for predicting 'Group' (Dementia vs. No Dementia)
    based on decision rules derived from a decision tree classifier (with interaction features).
    """
    def predict_group(self, data_point, provide_reasoning=False):
        """
        Predicts 'Group' (0: Nondemented, 1: Demented) using the derived interaction features.
        
        Args:
            data_point (dict): Dictionary containing both base and interaction features.
            provide_reasoning (bool): If True, returns a detailed reasoning path.
        
        Returns:
            tuple: (prediction, reasoning) where prediction is 0 or 1.
        """
        reasoning = []
        if provide_reasoning:
            reasoning.append("────────────────────────────────────────")
            reasoning.append("**Evaluate Reasoning with Interaction Features Expert System**")
            reasoning.append("────────────────────────────────────────")
            reasoning.append("Starting CDR evaluation process using interaction feature rules...")
            reasoning.append("")
            reasoning.append("**Step 1: Assessing MMSE Score**")
            reasoning.append(f"- MMSE = {data_point['MMSE']:.2f} (Threshold: 26.50)")
            if data_point['MMSE'] <= 26.50:
                reasoning.append("  → Low MMSE scores are strongly associated with cognitive decline.")
            else:
                reasoning.append("  → High MMSE scores generally indicate preserved cognitive function.")
            reasoning.append("")

        # Extract features
        mmse = data_point['MMSE']
        etiv_asf = data_point['eTIV_ASF']
        nwbv_mmse = data_point['nWBV_MMSE']
        age_mmse = data_point['Age_MMSE']
        dementia_risk_index = data_point['Dementia_Risk_Index']
        etiv = data_point['eTIV']
        gender_age = data_point['Gender_Age']
        educ = data_point['EDUC']
        gender_mmse = data_point['Gender_MMSE']
        asf = data_point['ASF']
        age_nwbv = data_point['Age_nWBV']
        ses_brain = data_point['SES_Brain']
        ses_mmse = data_point['SES_MMSE']
        ses = data_point['SES']
        neuro_risk = data_point['Neuro_Risk']
        ses_age = data_point['SES_Age']
        nwbv = data_point['nWBV']

        # Decision Rules with more detailed explanation
        if mmse <= 26.50:
            if provide_reasoning:
                reasoning.append("**Decision Path A: MMSE is LOW**")
                reasoning.append(f"- Since MMSE ({mmse:.2f}) is <= 26.50, the patient is at risk for cognitive impairment.")
            if etiv_asf <= 1754.99:
                if provide_reasoning:
                    reasoning.append(f"- eTIV_ASF = {etiv_asf:.2f} (<= 1754.99) reinforces a protective brain volume effect.")
                    reasoning.append("  → This combination supports a **Nondemented (0)** diagnosis.")
                return 0, reasoning
            else:  # etiv_asf > 1754.99
                if provide_reasoning:
                    reasoning.append(f"- eTIV_ASF = {etiv_asf:.2f} (> 1754.99) is higher than expected for low MMSE.")
                    reasoning.append("  → This suggests an abnormality in brain volume normalization leading to a **Demented (1)** classification.")
                return 1, reasoning
        else:  # mmse > 26.50
            if provide_reasoning:
                reasoning.append("**Decision Path B: MMSE is HIGH**")
                reasoning.append(f"- MMSE ({mmse:.2f}) > 26.50 indicates overall good cognitive performance.")
            if nwbv_mmse <= 22.04:
                if provide_reasoning:
                    reasoning.append(f"- nWBV_MMSE = {nwbv_mmse:.2f} (<= 22.04) raises concerns despite a high MMSE.")
                if age_mmse <= 2055.50:
                    if provide_reasoning:
                        reasoning.append(f"- Age_MMSE = {age_mmse:.2f} (<= 2055.50) suggests that the age-adjusted cognitive score is moderate.")
                    if etiv_asf <= 1755.01:
                        if provide_reasoning:
                            reasoning.append(f"- eTIV_ASF = {etiv_asf:.2f} (<= 1755.01) further indicates a reduced brain volume ratio.")
                        if etiv_asf <= 1755.00:
                            if provide_reasoning:
                                reasoning.append("  → Final decision: **Demented (1)**. The cumulative evidence from low nWBV_MMSE and low eTIV_ASF outweighs the high MMSE.")
                            return 1, reasoning
                        else:
                            if provide_reasoning:
                                reasoning.append("  → Final decision: **Nondemented (0)**. A marginally higher eTIV_ASF tilts the balance toward normal cognition.")
                            return 0, reasoning
                    else:
                        if provide_reasoning:
                            reasoning.append("  → eTIV_ASF is high (> 1755.01), which favors a **Nondemented (0)** outcome despite a low nWBV_MMSE.")
                        return 0, reasoning
                else:  # age_mmse > 2055.50
                    if provide_reasoning:
                        reasoning.append(f"- Age_MMSE = {age_mmse:.2f} (> 2055.50) calls for further evaluation of gender-specific factors.")
                    if gender_age <= 74.50:
                        if provide_reasoning:
                            reasoning.append(f"- Gender_Age = {gender_age:.2f} (<= 74.50) is within expected range.")
                        if nwbv_mmse <= 19.53:
                            if provide_reasoning:
                                reasoning.append(f"- nWBV_MMSE = {nwbv_mmse:.2f} (<= 19.53) is concerning.")
                            if educ <= 17.00:
                                if provide_reasoning:
                                    reasoning.append(f"- EDUC = {educ:.2f} (<= 17.00) further supports vulnerability.")
                                    reasoning.append("  → Final decision: **Demented (1)** due to compounded low metrics.")
                                return 1, reasoning
                            else:
                                if provide_reasoning:
                                    reasoning.append(f"- EDUC = {educ:.2f} (> 17.00) suggests some cognitive reserve.")
                                    reasoning.append("  → Final decision: **Nondemented (0)**, balancing the low nWBV_MMSE.")
                                return 0, reasoning
                        else:
                            if provide_reasoning:
                                reasoning.append(f"- nWBV_MMSE = {nwbv_mmse:.2f} (> 19.53) is less concerning.")
                            if asf <= 1.23:
                                if provide_reasoning:
                                    reasoning.append(f"- ASF = {asf:.2f} (<= 1.23) supports the current trend.")
                                if age_mmse <= 2108.50:
                                    if provide_reasoning:
                                        reasoning.append(f"- Age_MMSE = {age_mmse:.2f} (<= 2108.50) confirms the moderate profile.")
                                    if gender_mmse <= 28.50:
                                        if provide_reasoning:
                                            reasoning.append(f"- Gender_MMSE = {gender_mmse:.2f} (<= 28.50) strengthens the risk profile.")
                                            reasoning.append("  → Final decision: **Demented (1)** based on the detailed interplay of these metrics.")
                                        return 1, reasoning
                                    else:
                                        if provide_reasoning:
                                            reasoning.append(f"- Gender_MMSE = {gender_mmse:.2f} (> 28.50) lowers the risk slightly.")
                                            reasoning.append("  → Final decision: **Nondemented (0)**.")
                                        return 0, reasoning
                                else:
                                    if provide_reasoning:
                                        reasoning.append(f"- Age_MMSE = {age_mmse:.2f} (> 2108.50) suggests additional caution.")
                                    if ses_brain <= 5.75:
                                        if provide_reasoning:
                                            reasoning.append(f"- SES_Brain = {ses_brain:.2f} (<= 5.75) is suboptimal.")
                                            reasoning.append("  → Final decision: **Nondemented (0)**.")
                                        return 0, reasoning
                                    else:
                                        if provide_reasoning:
                                            reasoning.append(f"- SES_Brain = {ses_brain:.2f} (> 5.75) indicates some risk.")
                                        if ses <= 3.50:
                                            if provide_reasoning:
                                                reasoning.append(f"- SES = {ses:.2f} (<= 3.50) further supports vulnerability.")
                                                reasoning.append("  → Final decision: **Demented (1)**.")
                                            return 1, reasoning
                                        else:
                                            if provide_reasoning:
                                                reasoning.append(f"- SES = {ses:.2f} (> 3.50) adds a protective factor.")
                                                reasoning.append("  → Final decision: **Nondemented (0)**.")
                                            return 0, reasoning
                            else:
                                if provide_reasoning:
                                    reasoning.append(f"- ASF = {asf:.2f} (> 1.23) alters the risk dynamics.")
                                if etiv_asf <= 1755.00:
                                    if provide_reasoning:
                                        reasoning.append(f"- eTIV_ASF = {etiv_asf:.2f} (<= 1755.00) supports a lower brain volume ratio.")
                                    if asf <= 1.27:
                                        if provide_reasoning:
                                            reasoning.append(f"- ASF remains controlled at {asf:.2f} (<= 1.27).")
                                        if nwbv_mmse <= 19.63:
                                            if provide_reasoning:
                                                reasoning.append(f"- nWBV_MMSE = {nwbv_mmse:.2f} (<= 19.63) is still a red flag.")
                                                reasoning.append("  → Final decision: **Nondemented (0)**.")
                                            return 0, reasoning
                                        else:
                                            if provide_reasoning:
                                                reasoning.append(f"- nWBV_MMSE = {nwbv_mmse:.2f} (> 19.63) tips the scale.")
                                                reasoning.append("  → Final decision: **Demented (1)**.")
                                            return 1, reasoning
                                    else:
                                        if provide_reasoning:
                                            reasoning.append(f"- ASF is higher at {asf:.2f} (> 1.27).")
                                        if age_nwbv <= 55.64:
                                            if provide_reasoning:
                                                reasoning.append(f"- Age_nWBV = {age_nwbv:.2f} (<= 55.64) is critical.")
                                                reasoning.append("  → Final decision: **Demented (1)**.")
                                            return 1, reasoning
                                        else:
                                            if provide_reasoning:
                                                reasoning.append(f"- Age_nWBV = {age_nwbv:.2f} (> 55.64) offers some reassurance.")
                                            if asf <= 1.38:
                                                if provide_reasoning:
                                                    reasoning.append(f"- ASF = {asf:.2f} (<= 1.38) further lowers risk.")
                                                    reasoning.append("  → Final decision: **Nondemented (0)**.")
                                                return 0, reasoning
                                            else:
                                                if provide_reasoning:
                                                    reasoning.append(f"- ASF = {asf:.2f} (> 1.38) adds risk.")
                                                    reasoning.append("  → Final decision: **Nondemented (0)** by default (based on Dementia_Risk_Index).")
                                                return 0, reasoning
                    else:
                        if provide_reasoning:
                            reasoning.append(f"- Gender_Age = {gender_age:.2f} (> 74.50): Redirecting evaluation.")
                        if etiv <= 1617.18:
                            if provide_reasoning:
                                reasoning.append(f"- eTIV = {etiv:.2f} (<= 1617.18) suggests a smaller brain volume.")
                            if age_nwbv <= 55.44:
                                if provide_reasoning:
                                    reasoning.append(f"- Age_nWBV = {age_nwbv:.2f} (<= 55.44) is within concerning limits.")
                                if etiv_asf <= 1755.00:
                                    if provide_reasoning:
                                        reasoning.append("  → Final decision: **Nondemented (0)**.")
                                    return 0, reasoning
                                else:
                                    if provide_reasoning:
                                        reasoning.append("  → Final decision: **Demented (1)**.")
                                    return 1, reasoning
                            else:
                                if provide_reasoning:
                                    reasoning.append(f"- Age_nWBV = {age_nwbv:.2f} (> 55.44): Indicates vulnerability.")
                                    reasoning.append("  → Final decision: **Demented (1)**.")
                                return 1, reasoning
                        else:
                            if provide_reasoning:
                                reasoning.append(f"- eTIV = {etiv:.2f} (> 1617.18): Suggests a larger intracranial volume.")
                            if ses_age <= 83.00:
                                if provide_reasoning:
                                    reasoning.append(f"- SES_Age = {ses_age:.2f} (<= 83.00) indicates moderate SES influence.")
                                if nwbv <= 0.72:
                                    if provide_reasoning:
                                        reasoning.append(f"- nWBV = {nwbv:.2f} (<= 0.72) is critically low.")
                                        reasoning.append("  → Final decision: **Demented (1)**.")
                                    return 1, reasoning
                                else:
                                    if provide_reasoning:
                                        reasoning.append(f"- nWBV = {nwbv:.2f} (> 0.72): Indicates better brain volume.")
                                    if etiv <= 1652.22:
                                        if provide_reasoning:
                                            reasoning.append(f"- eTIV = {etiv:.2f} (<= 1652.22) supports the risk profile.")
                                            reasoning.append("  → Final decision: **Demented (1)**.")
                                        return 1, reasoning
                                    else:
                                        if provide_reasoning:
                                            reasoning.append(f"- eTIV = {etiv:.2f} (> 1652.22) is protective.")
                                            reasoning.append("  → Final decision: **Nondemented (0)**.")
                                        return 0, reasoning
                            else:
                                if provide_reasoning:
                                    reasoning.append(f"- SES_Age = {ses_age:.2f} (> 83.00) indicates high SES-age interaction.")
                                    reasoning.append("  → Final decision: **Nondemented (0)**.")
                                return 0, reasoning
            else:
                if provide_reasoning:
                    reasoning.append("**Decision Path C: High nWBV_MMSE**")
                    reasoning.append(f"- nWBV_MMSE = {nwbv_mmse:.2f} (> 22.04) is generally favorable.")
                if ses_mmse <= 29.50:
                    if provide_reasoning:
                        reasoning.append(f"- SES_MMSE = {ses_mmse:.2f} (<= 29.50) raises concern.")
                        reasoning.append("  → Final decision: **Demented (1)**.")
                    return 1, reasoning
                else:
                    if provide_reasoning:
                        reasoning.append(f"- SES_MMSE = {ses_mmse:.2f} (> 29.50) is reassuring.")
                    if etiv_asf <= 1755.01:
                        if provide_reasoning:
                            reasoning.append(f"- eTIV_ASF = {etiv_asf:.2f} (<= 1755.01) indicates risk.")
                        if neuro_risk <= 0.03:
                            if provide_reasoning:
                                reasoning.append(f"- Neuro_Risk = {neuro_risk:.2f} (<= 0.03) is low.")
                                reasoning.append("  → Final decision: **Nondemented (0)**.")
                            return 0, reasoning
                        else:
                            if provide_reasoning:
                                reasoning.append(f"- Neuro_Risk = {neuro_risk:.2f} (> 0.03) is concerning.")
                            if ses_age <= 69.50:
                                if provide_reasoning:
                                    reasoning.append(f"- SES_Age = {ses_age:.2f} (<= 69.50) adds to the risk.")
                                if gender_age <= 34.00:
                                    if provide_reasoning:
                                        reasoning.append(f"- Gender_Age = {gender_age:.2f} (<= 34.00) further supports vulnerability.")
                                        reasoning.append("  → Final decision: **Demented (1)**.")
                                    return 1, reasoning
                                else:
                                    if provide_reasoning:
                                        reasoning.append(f"- Gender_Age = {gender_age:.2f} (> 34.00) mitigates risk slightly.")
                                        reasoning.append("  → Final decision: **Nondemented (0)**.")
                                    return 0, reasoning
                            else:
                                if provide_reasoning:
                                    reasoning.append(f"- SES_Age = {ses_age:.2f} (> 69.50) is protective.")
                                    reasoning.append("  → Final decision: **Nondemented (0)**.")
                                return 0, reasoning
                    else:
                        if provide_reasoning:
                            reasoning.append(f"- eTIV_ASF = {etiv_asf:.2f} (> 1755.01) supports good brain structure.")
                        if neuro_risk <= 0.03:
                            if provide_reasoning:
                                reasoning.append(f"- Neuro_Risk = {neuro_risk:.2f} (<= 0.03) is low, leading to a **Nondemented (0)** outcome.")
                            return 0, reasoning
                        else:
                            if provide_reasoning:
                                reasoning.append(f"- Neuro_Risk = {neuro_risk:.2f} (> 0.03) is elevated, tipping the scale to **Demented (1)**.")
                            return 1, reasoning

        # Fallback (should never be reached)
        return 0, ["No rule matched"] if provide_reasoning else 0

    def format_data_point(self, input_data):
        """
        In this setup, the input data is already a dictionary with both base and derived features.
        """
        return input_data

# ---------------------------
# Streamlit App Interface
# ---------------------------
def main():
    st.title("🧠 Dementia Prediction Expert System")
    st.markdown("Enter the patient’s **base** details below. All additional interaction features will be computed automatically.")
    
    with st.form(key='input_form'):
        col1, col2 = st.columns(2)
        with col1:
            mmse = st.number_input("MMSE", min_value=0.0, max_value=30.0, value=27.0, step=0.1)
            age = st.number_input("Age", min_value=0.0, max_value=120.0, value=70.0, step=0.1)
            asf = st.number_input("ASF", min_value=0.0, value=1.0, step=0.01)
            nwbv = st.number_input("nWBV", min_value=0.0, max_value=1.0, value=0.7, step=0.001)
        with col2:
            gender = st.selectbox("Gender", options=["Male", "Female"])
            educ = st.number_input("EDUC", min_value=0.0, max_value=20.0, value=16.0, step=0.1)
            ses = st.number_input("SES", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
            etiv = st.number_input("eTIV", min_value=0.0, value=1800.0, step=0.1)
        
        provide_reasoning = st.checkbox("Show Detailed Reasoning", value=True)
        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        # Map gender to numeric (Male=1, Female=0)
        mf_value = 1 if gender == "Male" else 0
        
        # Base features from user input
        base_input = {
            'MMSE': mmse,
            'Age': age,
            'ASF': asf,
            'nWBV': nwbv,
            'M/F': mf_value,
            'EDUC': educ,
            'SES': ses,
            'eTIV': etiv
        }
        
        # Compute derived interaction features
        complete_input = compute_interaction_features(base_input)
        
        # Additional features needed by the expert system (if not already computed)
        complete_input['nWBV_MMSE'] = complete_input['nWBV'] * complete_input['MMSE']
        complete_input['Age_MMSE'] = complete_input['Age'] * complete_input['MMSE']
        complete_input['Age_nWBV'] = complete_input['Age'] * complete_input['nWBV']
        complete_input['SES_MMSE'] = complete_input['SES'] * complete_input['MMSE']
        
        # Display the computed new features in a neat table
        st.markdown("## Calculated Interaction Features")
        # Exclude the base features for clarity
        derived_keys = [key for key in complete_input.keys() if key not in ['MMSE', 'Age', 'ASF', 'nWBV', 'M/F', 'EDUC', 'SES', 'eTIV']]
        derived_features = {key: complete_input[key] for key in derived_keys}
        df_features = pd.DataFrame.from_dict(derived_features, orient='index', columns=["Value"])
        st.table(df_features)
        
        st.markdown("---")
        
        # Run expert system prediction
        expert_system = ExpertSystem_Group_InteractionFeatures_Classification()
        prediction, reasoning = expert_system.predict_group(complete_input, provide_reasoning=provide_reasoning)
        
        # Display the prediction result
        st.markdown("## Prediction Result")
        diagnosis = "Demented (1)" if prediction == 1 else "Nondemented (0)"
        if prediction == 1:
            st.error(f"**Diagnosis:** {diagnosis}")
        else:
            st.success(f"**Diagnosis:** {diagnosis}")
        
        st.markdown("---")
        st.markdown("### Detailed Reasoning Path")
        for line in reasoning:
            st.markdown(f"* {line}")

if __name__ == '__main__':
    main()
