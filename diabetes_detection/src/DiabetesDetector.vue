<template>
  <div id="app">
    <Header />
    <Banner :currentComponent="currentComponent" @update-component="updateComponent" />
    <!-- Content -->
    <div class="content">
      <component :is="currentComponent"></component>
    </div>
  </div>
</template>

<script>
import Header from './components/Header.vue'
import Banner from './components/Banner.vue'
import Appointment from './components/Appointment.vue';
import Symptoms from './components/Symptoms.vue';
import Diagnosis from './components/Diagnosis.vue';

export default {
  name: 'App',
  components: {
    Header,
    Banner,
    Appointment,
    Symptoms,
    Diagnosis
  },
  data() {
    return {
      currentComponent: 'Symptoms' // 默认显示Symptoms组件
    };
  },methods: {
    updateComponent(componentName) {
      this.currentComponent = componentName;
    }
  }
}
</script>

<style>
#app {
  
  text-align: center;
 
}
</style>