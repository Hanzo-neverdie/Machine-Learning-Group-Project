<template>
  <div class="header">
    <h2>Diabetes Prediction Form</h2>
    <p>Please enter some of your personal health data below. This information will help us predict your risk of diabetes.</p>
  </div>
  <div class="form-container">
    
    <form @submit.prevent="submitForm">
      <!-- Pregnancies and Glucose -->
      <div class="form-row">
        <div class="form-group">
          <label for="pregnancies">Pregnancies</label>
          <input
            type="number"
            id="pregnancies"
            v-model.number="formData.Pregnancies"
            required
          />
        </div>
        <div class="form-group">
          <label for="glucose">Glucose</label>
          <input
            type="number"
            id="glucose"
            v-model.number="formData.Glucose"
            required
          />
        </div>
      </div>

      <!-- BloodPressure and SkinThickness -->
      <div class="form-row">
        <div class="form-group">
          <label for="bloodPressure">Blood Pressure</label>
          <input
            type="number"
            id="bloodPressure"
            v-model.number="formData.BloodPressure"
            required
          />
        </div>
        <div class="form-group">
          <label for="skinThickness">Skin Thickness</label>
          <input
            type="number"
            id="skinThickness"
            v-model.number="formData.SkinThickness"
            required
          />
        </div>
      </div>

      <!-- Insulin and BMI -->
      <div class="form-row">
        <div class="form-group">
          <label for="insulin">Insulin</label>
          <input
            type="number"
            id="insulin"
            v-model.number="formData.Insulin"
            required
          />
        </div>
        <div class="form-group">
          <label for="bmi">BMI</label>
          <input
            type="number"
            id="bmi"
            v-model.number="formData.BMI"
            step="0.1"
            required
          />
        </div>
      </div>

      <!-- DiabetesPedigreeFunction and Age -->
      <div class="form-row">
        <div class="form-group">
          <label for="diabetesPedigreeFunction">Diabetes Pedigree Function</label>
          <input
            type="number"
            id="diabetesPedigreeFunction"
            v-model.number="formData.DiabetesPedigreeFunction"
            step="0.001"
            required
          />
        </div>
        <div class="form-group">
          <label for="age">Age</label>
          <input
            type="number"
            id="age"
            v-model.number="formData.Age"
            required
          />
        </div>
      </div>

      <button type="submit">Submit</button>
    </form>
    <div v-if="predictionResult">
      <h3 style="color: #007bff; text-align: center; margin-top: 20px;">Prediction Result:</h3>
      <p style="text-align: center; font-size: larger;" :style="predictionColor">{{ predictionResult }}</p>
      <p style="text-align: center;">Probability: {{ predictionProbability.toFixed(3) }}</p>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      formData: {
        Pregnancies: null,
        Glucose: null,
        BloodPressure: null,
        SkinThickness: null,
        Insulin: null,
        BMI: null,
        DiabetesPedigreeFunction: null,
        Age: null
      },
      predictionResult: null,
      predictionProbability: null
    };
  },
  computed: {
    predictionColor() {
      
      if (this.predictionResult === 'Diabetic') {
        return { color: 'red' };
      } else if (this.predictionResult === 'Non-Diabetic') {
        return { color: 'green' };
      } else {
        return { color: 'black' }; // 默认颜色
      }
    }
  },
  methods: {
    async submitForm() {
      try {
        // 显示加载状态
        this.$emit('loading', true);

        // 发送 POST 请求到后端
        const response = await axios.post('http://localhost:5000/predict', this.formData);

        // 请求成功，处理响应数据
        console.log('Response:', response.data);
        this.predictionResult = response.data.result;
        this.predictionProbability = response.data.probability;

        // 重置表单
        this.formData = {
          Pregnancies: null,
          Glucose: null,
          BloodPressure: null,
          SkinThickness: null,
          Insulin: null,
          BMI: null,
          DiabetesPedigreeFunction: null,
          Age: null
        };
      } catch (error) {
        // 请求失败，处理错误
        console.error('Error submitting form:', error);
        alert('Failed to submit form. Please try again.');
      } finally {
        // 隐藏加载状态
        this.$emit('loading', false);
      }
    }
  }
};
</script>

<style scoped>
.form-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  border: 0px solid #ccc;
  border-radius: 5px;
  background-color: white;
}
.header {
  text-align: center;
  margin-top: 20px;
}
h2 {
  text-align: center;
  margin-bottom: 20px;
}

.form-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 15px;
}

.form-group {
  flex: 1;
  margin-right: 15px; /* 添加一些间距 */
}

.form-group:last-child {
  margin-right: 0; /* 最后一个输入框不需要右边距 */
}

label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

input[type="number"] {
  width: 100%;
  padding: 8px;
  box-sizing: border-box;
  border-radius: 5px;
}

button {
  width: 100%;
  padding: 10px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
}

button:hover {
  background-color: #0056b3;
}
</style>