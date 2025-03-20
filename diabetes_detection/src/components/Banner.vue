<template>
    <div class="banner">
        <a href="#"><h1>Diabetes</h1></a>
      <div class="buttons">
        <!-- 使用Vue Router的<router-link> -->
        <a  @click="updateComponent('Appointment')" class="button"><img src="../assets/calender.svg" alt="">
            Request an Appointment</a>
      </div>
      <hr>
      <div id="app"><div class="options">
        <a  @click="updateComponent('Symptoms')" 
        :class="{ active: currentComponent === 'Symptoms', inactive: currentComponent !== 'Symptoms' }" class="option">Symptoms & causes</a>
        <a  @click="updateComponent('Diagnosis')"
        :class="{ active: currentComponent === 'Diagnosis', inactive: currentComponent !== 'Diagnosis' }" class="option">Diagnosis & treatment</a>
        </div>
      </div>
      
    </div>
  </template>
<script>
export default {
  props: {
    currentComponent: String
  },
  methods: {
    updateComponent(componentName) {
      this.$emit('update-component', componentName);
    }
  }
};
</script>
  <style scoped>
  .banner {
    background: linear-gradient(to right, #316aa7, #00a4ff); 
    color: white;
    padding: 50px 20px;
    text-align: center;
    
  }
  .banner a {
       
        color: white;
    }
    .button {
        cursor: pointer; /* 显示可点击的光标 */
       
        color: white; /* 默认颜色 */
    
    }
  .banner h1 {
    margin: 0;
    font-size: 36px;
  }
  
  .banner .buttons {
    margin-top: 20px;
    
  }
  .banner .buttons img {
    margin-top: -10px;
    height: 30px;
    margin-right: 10px;
  }
    .banner .buttons a {
        color: white;
       
    }
    
  .banner hr{
    margin-top: 30px;
    border: 1px solid rgb(110, 159, 211);
  }
 
 
  .options {
    display: flex;
    margin-left: 35%;
  }
  .option {
    cursor: pointer;
    margin-left: 20px;
    background-color: #00a4ff;
    padding: 10px 20px;
    text-decoration: none;
    border-radius: 15px;
  }
  .option:hover {
  background-color: #ffffff; /* 悬停时的背景颜色 */
  color: #007bff; /* 悬停时的字体颜色 */
}
.option.active {
  background-color: #ffffff; /* 激活状态的背景颜色 */
  color: #007bff; /* 激活状态的字体颜色 */
}

  </style>