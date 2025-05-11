import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Dự đoán kết quả học tập", layout="centered")
st.markdown(
    "<h3 style='color: blue;'>DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN</h3>",
    unsafe_allow_html=True
)

# =============================
# 1. Sidebar: Chọn loại sinh viên và nhập dữ liệu
# =============================
st.sidebar.subheader("Cài đặt đầu vào")
student_type = st.sidebar.selectbox("Loại sinh viên:", ("8 kỳ", "10 kỳ"))
# Đặt max kỳ để có thể dự đoán kỳ tiếp theo (8 kỳ học thực chất có 7 kỳ để dự GPA kỳ 8)
max_semester = 7 if student_type == "8 kỳ" else 9
selectable_semesters = list(range(1, max_semester + 1))
current_semester = st.sidebar.selectbox("Kỳ hiện tại:", selectable_semesters)

# Nhập GPA và Tín chỉ cho các kỳ đã chọn
gpa_inputs = []
credit_inputs = []
for i in range(1, current_semester + 1):
    gpa = st.sidebar.number_input(f"GPA kỳ {i}", min_value=0.0, max_value=4.0, step=0.01, format="%.2f")
    credit = st.sidebar.number_input(f"Tín chỉ kỳ {i}", min_value=1, max_value=28, step=1)
    gpa_inputs.append(gpa)
    credit_inputs.append(credit)

# =============================
# 2. Kiểm tra dữ liệu đầu vào
# =============================
if len(gpa_inputs) != current_semester or len(credit_inputs) != current_semester:
    st.warning("⚠️ Vui lòng nhập đủ GPA và tín chỉ cho các kỳ đã chọn.")
else:
    try:
        # Chuẩn bị feature vector: [GPA_1,...,GPA_n, TC_1,...,TC_n]
        features = gpa_inputs + credit_inputs
        X = np.array(features).reshape(1, -1)

        # Xác định key cho dict model: GPA_TC_1, GPA_TC_1_2, ...
        semesters = list(range(1, current_semester + 1))
        key = 'GPA_TC_' + '_'.join(map(str, semesters))

        # Prefix cho file model: '8' hoặc '10'
        prefix = '8' if student_type == '8 kỳ' else '10'

        # =============================
        # Dự đoán Final CPA
        # =============================
        path_cpa = f"models_streamlit/final_cpa_tc_{prefix}_ki.joblib"
        dict_cpa = joblib.load(path_cpa)
        scaler_cpa = dict_cpa[key]['scaler']
        model_cpa  = dict_cpa[key]['model']
        Xc = scaler_cpa.transform(X)
        cpa_pred = model_cpa.predict(Xc)[0]

        st.subheader("🎓 Dự đoán CPA tốt nghiệp")
        st.success(f"Final CPA: {cpa_pred:.2f}")

        # =============================
        # Dự đoán GPA Kỳ tiếp theo
        # =============================
        # Với student_type 8 kỳ: max_semester=7 => kỳ 8 dự
        # Với 10 kỳ: max_semester=9 => kỳ 10 dự
        if current_semester < max_semester + 1:
            path_gpa = f"models_streamlit/next_gpa_tc_{prefix}_ki.joblib"
            dict_gpa = joblib.load(path_gpa)
            scaler_gpa = dict_gpa[key]['scaler']
            model_gpa  = dict_gpa[key]['model']
            Xg = scaler_gpa.transform(X)
            gpa_pred = model_gpa.predict(Xg)[0]

            next_sem = current_semester + 1
            st.subheader(f"📘 Dự đoán GPA kỳ {next_sem}")
            st.info(f"GPA dự đoán: {gpa_pred:.2f}")

    except KeyError as ke:
        avail = ', '.join(dict_cpa.keys())
        st.error(f"❌ Key '{{ke.args[0]}}' không tồn tại. Các key khả dụng: {avail}")
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {e}")
