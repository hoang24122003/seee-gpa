import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Dự đoán kết quả học tập", layout="centered")
st.markdown(
    "<h3 style='color: blue;'>DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN</h3>",
    unsafe_allow_html=True
)

# =============================
# 1. Sidebar chọn loại sinh viên
# =============================
st.sidebar.subheader("Cài đặt đầu vào")
student_type = st.sidebar.selectbox("Loại sinh viên:", ("8 kỳ", "10 kỳ"))
max_semester = 6 if student_type == "8 kỳ" else 8

# Chọn kỳ hiện tại
current_semester = st.sidebar.selectbox("Kỳ hiện tại:", list(range(1, max_semester + 1)))

# Nhập GPA và tín chỉ cho từng kỳ
gpa_inputs = []
tc_inputs = []
for i in range(1, current_semester + 1):
    gpa = st.sidebar.number_input(f"GPA kỳ {i}", min_value=0.0, max_value=4.0, step=0.01, format="%.2f")
    tc = st.sidebar.number_input(f"Tín chỉ kỳ {i}", min_value=1, max_value=28, step=1, format="%d")
    gpa_inputs.append(gpa)
    tc_inputs.append(tc)

# =============================
# 2. Kiểm tra và xử lý dữ liệu đầu vào
# =============================
if any(g == 0.0 for g in gpa_inputs) or any(tc == 0 for tc in tc_inputs):
    st.warning("⚠️ Vui lòng nhập đầy đủ GPA và Tín chỉ cho tất cả các kỳ đã chọn.")
else:
    try:
        # Gộp GPA và TC theo thứ tự [GPA_1, TC_1, GPA_2, TC_2, ...]
        combined_inputs = []
        for g, t in zip(gpa_inputs, tc_inputs):
            combined_inputs.extend([g, t])
        input_data = np.array(combined_inputs).reshape(1, -1)

        model_prefix = student_type.split()[0]  # '8' hoặc '10'

        # =============================
        # 3. Dự đoán Final CPA
        # =============================
        group_key_cpa = f"GPA_TC_1_{current_semester}" if current_semester > 1 else "GPA_TC_1"
        cpa_model_path = f"models_streamlit/final_cpa_tc_{model_prefix}_ki.joblib"
        cpa_dict = joblib.load(cpa_model_path)

        if group_key_cpa not in cpa_dict:
            st.error(f"❌ Key '{group_key_cpa}' không tồn tại trong mô hình CPA.")
        else:
            scaler_cpa = cpa_dict[group_key_cpa]['scaler']
            model_cpa = cpa_dict[group_key_cpa]['model']
            expected_len = len(scaler_cpa.mean_)
            if input_data.shape[1] != expected_len:
                st.error(f"❌ Final CPA: Số đặc trưng ({input_data.shape[1]}) không khớp mô hình ({expected_len})")
            else:
                predicted_cpa = model_cpa.predict(scaler_cpa.transform(input_data))[0]
                st.subheader("🎓 Dự đoán CPA tốt nghiệp:")
                st.success(f"Final CPA: {predicted_cpa:.2f}")

        # =============================
        # 4. Dự đoán GPA kỳ tiếp theo
        # =============================
        if current_semester < max_semester:
            group_key_gpa = f"GPA_{current_semester + 1}"  # Chú ý: đúng key theo file joblib đã lưu
            next_gpa_path = f"models_streamlit/next_gpa_tc_{model_prefix}_ki.joblib"
            next_dict = joblib.load(next_gpa_path)

            if group_key_gpa not in next_dict:
                st.error(f"❌ Key '{group_key_gpa}' không tồn tại trong mô hình GPA.")
            else:
                scaler_next = next_dict[group_key_gpa]['scaler']
                model_next = next_dict[group_key_gpa]['model']
                expected_len_next = len(scaler_next.mean_)
                if input_data.shape[1] != expected_len_next:
                    st.error(f"❌ GPA kỳ tiếp theo: Số đặc trưng ({input_data.shape[1]}) không khớp mô hình ({expected_len_next})")
                else:
                    predicted_next_gpa = model_next.predict(scaler_next.transform(input_data))[0]
                    st.subheader(f"📘 Dự đoán GPA kỳ {current_semester + 1}:")
                    st.info(f"GPA dự đoán: {predicted_next_gpa:.2f}")

    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi khi dự đoán: {e}")
