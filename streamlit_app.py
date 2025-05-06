import streamlit as st
import numpy as np
import joblib

# ========================
# 1. Giao diện
# ========================

st.header(":blue[DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN]")

# Sidebar chọn loại sinh viên
st.sidebar.subheader("Chọn loại sinh viên")
student_type = st.sidebar.selectbox("Loại sinh viên:", ("8 kỳ", "10 kỳ"))

# Xác định số kỳ tối đa theo loại sinh viên
max_semester = 6 if student_type == "8 kỳ" else 8

# Chọn kỳ hiện tại
current_semester = st.sidebar.selectbox("Chọn kỳ hiện tại:", list(range(1, max_semester + 1)))

# Nhập GPA từng kỳ
gpa_inputs = []
for i in range(1, current_semester + 1):
    gpa = st.sidebar.number_input(f"GPA kỳ {i}", min_value=0.0, max_value=4.0, step=0.01, format="%.2f")
    gpa_inputs.append(gpa)

# ========================
# 2. Xử lý dữ liệu đầu vào
# ========================

if any(g == 0.0 for g in gpa_inputs):
    st.warning("Vui lòng nhập đầy đủ GPA cho tất cả các kỳ đã chọn!")
else:
    try:
        # Chuẩn hóa input
        input_data = np.array(gpa_inputs).reshape(1, -1)

        # ========================
        # 3. Dự đoán Final CPA
        # ========================
        model_final_cpa_path = f"models_streamlit/final_cpa_{student_type.split()[0]}_ki.joblib"
        model_final = joblib.load(model_final_cpa_path)

        predicted_cpa = model_final.predict(input_data)[0]
        st.subheader("🎓 Dự đoán CPA tốt nghiệp:")
        st.success(f"Final CPA dự đoán: {predicted_cpa:.2f}")

        # ========================
        # 4. Dự đoán GPA kỳ tiếp theo
        # ========================
        if current_semester < max_semester:
            next_model_path = f"models_streamlit/next_gpa_{student_type.split()[0]}_ki.joblib"
            model_next_gpa = joblib.load(next_model_path)

            next_gpa = model_next_gpa.predict(input_data)[0]
            st.subheader(f"📘 Dự đoán GPA kỳ {current_semester + 1}:")
            st.info(f"GPA kỳ {current_semester + 1} dự đoán: {next_gpa:.2f}")

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi dự đoán: {str(e)}")
