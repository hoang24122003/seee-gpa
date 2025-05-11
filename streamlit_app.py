import streamlit as st
import numpy as np
import joblib

# Cấu hình trang
st.set_page_config(page_title="Dự đoán kết quả học tập", layout="centered")
st.markdown(
    "<h3 style='color: blue;'>DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN</h3>",
    unsafe_allow_html=True
)

# =============================
# 1. Sidebar - Lấy đầu vào từ người dùng
# =============================
st.sidebar.subheader("Cài đặt đầu vào")
# Chọn loại sinh viên (8 kỳ hoặc 10 kỳ)
student_type = st.sidebar.selectbox("Loại sinh viên:", ("8 kỳ", "10 kỳ"))
max_semester = 6 if student_type == "8 kỳ" else 8

# Chọn kỳ hiện tại để dự đoán
current_semester = st.sidebar.selectbox(
    "Kỳ hiện tại:", list(range(1, max_semester + 1))
)

# Nhập GPA và tín chỉ từng kỳ
st.sidebar.markdown("---")
gpa_inputs = []
tc_inputs = []
for i in range(1, current_semester + 1):
    gpa = st.sidebar.number_input(
        label=f"GPA kỳ {i}",
        min_value=0.0,
        max_value=4.0,
        step=0.01,
        format="%.2f",
        key=f"gpa_{i}"
    )
    tc = st.sidebar.number_input(
        label=f"Tín chỉ kỳ {i}",
        min_value=0,
        max_value=30,
        step=1,
        key=f"tc_{i}"
    )
    gpa_inputs.append(gpa)
    tc_inputs.append(tc)

# Kiểm tra nhập liệu hợp lệ (cảnh báo nếu còn giá trị mặc định)
if any(g == 0.0 for g in gpa_inputs) or any(t == 0 for t in tc_inputs):
    st.sidebar.warning("⚠️ Vui lòng nhập đầy đủ GPA và tín chỉ cho tất cả các kỳ đã chọn.")
else:
    # =============================
    # 2. Xử lý dữ liệu và dự đoán
    # =============================
    try:
        # Tạo feature vector
        input_features = np.array(gpa_inputs + tc_inputs).reshape(1, -1)

        # Sinh group_key như trong file model: "GPA_TC_1", "GPA_TC_1_2", ...
        semesters = [str(i) for i in range(1, current_semester + 1)]
        group_key = "GPA_TC_" + "_".join(semesters)

        # Prefix để định danh folder/model (8 hoặc 10)
        model_prefix = student_type.split()[0]

        # ----- Dự đoán Final CPA -----
        cpa_model_path = f"models_streamlit/final_cpa_tc_{model_prefix}_ki.joblib"
        cpa_dict = joblib.load(cpa_model_path)
        scaler_cpa = cpa_dict[group_key]['scaler']
        model_cpa  = cpa_dict[group_key]['model']
        # Scale và dự đoán
        input_scaled_cpa = scaler_cpa.transform(input_features)
        predicted_cpa    = model_cpa.predict(input_scaled_cpa)[0]

        st.subheader("🎓 Dự đoán CPA tốt nghiệp")
        st.success(f"Final CPA: {predicted_cpa:.2f}")

        # ----- Dự đoán GPA kỳ tiếp theo -----
        if current_semester < max_semester:
            next_gpa_path = f"models_streamlit/next_gpa_tc_{model_prefix}_ki.joblib"
            next_dict = joblib.load(next_gpa_path)
            scaler_next = next_dict[group_key]['scaler']
            model_next  = next_dict[group_key]['model']

            input_scaled_next = scaler_next.transform(input_features)
            predicted_next_gpa = model_next.predict(input_scaled_next)[0]

            st.subheader(f"📘 Dự đoán GPA kỳ {current_semester + 1}")
            st.info(f"GPA dự đoán: {predicted_next_gpa:.2f}")

    except KeyError as ke:
        available = ', '.join(cpa_dict.keys())
        st.error(
            f"❌ Key '{ke.args[0]}' không tồn tại trong model. "
            f"Các key khả dụng: {available}"
        )
    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi khi dự đoán: {e}")
