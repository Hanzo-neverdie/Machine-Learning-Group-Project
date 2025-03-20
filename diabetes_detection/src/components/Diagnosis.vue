<template>
    <div id="app">
      
      <div class="container">
        <div class="content">
          <h2>Diagnosis</h2>
          <p style="color: grey; font-weight: bold;">Type 1 diabetes FAQs</p>
          <p style="color: gray;">Endocrinologist Yogish Kudva, M.B.B.S., answers the most frequently asked questions about type 1 diabetes.</p>
          <div class="video-container">       
            <video controls>
                <source src="https://youtu.be/CwxStExoyGA" type="video/WebM">
                Your browser does not support the video tag.
            </video>
          </div>
          <p>Type 1 diabetes symptoms often start suddenly and are often the reason for checking blood sugar levels. Because symptoms of other types of diabetes and prediabetes come on more gradually or may not be easy to see, the American Diabetes Association (ADA) has developed screening guidelines. The ADA recommends that the following people be screened for diabetes:</p>
          <ul>
            <li> <b>Anyone with a body mass index higher than 25 (23 for Asian Americans), regardless of age, </b> who has additional risk factors. These factors include high blood pressure, non-typical cholesterol levels, an inactive lifestyle, a history of polycystic ovary syndrome or heart disease, and having a close relative with diabetes. </li>
            <li> <b>Anyone older than age 35</b> is advised to get an initial blood sugar screening. If the results are normal, they should be screened every three years after that.</li>
            <li> <b>Women who have had gestational diabetes</b>  are advised to be screened for diabetes every three years.</li>
            <li> <b>Anyone who has been diagnosed with prediabetes</b>  is advised to be tested every year.</li>
            <li> <b>Anyone who has HIV </b> is advised to be tested.</li>
          </ul>
        </div>
        <hr>
        <div class="content">
          <h2>Treatment</h2>
          <p>Depending on what type of diabetes you have, blood sugar monitoring, insulin and oral drugs may be part of your treatment. Eating a healthy diet, staying at a healthy weight and getting regular physical activity also are important parts of managing diabetes.</p>
          <p style="font-weight: bolder; font-size: large;">Treatments for all types of diabetes</p>
          <p>An important part of managing diabetes — as well as your overall health — is keeping a healthy weight through a healthy diet and exercise plan:</p>
          <ul>
            <li> <b>Healthy eating.</b> Your diabetes diet is simply a healthy-eating plan that will help you control your blood sugar. You'll need to focus your diet on more fruits, vegetables, lean proteins and whole grains. These are foods that are high in nutrition and fiber and low in fat and calories. You'll also cut down on saturated fats, refined carbohydrates and sweets. In fact, it's the best eating plan for the entire family. Sugary foods are OK once in a while. They must be counted as part of your meal plan.</li>
            <p>
            Understanding what and how much to eat can be a challenge. A registered dietitian can help you create a meal plan that fits your health goals, food preferences and lifestyle. This will likely include carbohydrate counting, especially if you have type 1 diabetes or use insulin as part of your treatment.</p>
            <li> <b>Physical activity.</b>  Everyone needs regular aerobic activity. This includes people who have diabetes. Physical activity lowers your blood sugar level by moving sugar into your cells, where it's used for energy. Physical activity also makes your body more sensitive to insulin. That means your body needs less insulin to transport sugar to your cells.</li>
            <p>Get your provider's OK to exercise. Then choose activities you enjoy, such as walking, swimming or biking. What's most important is making physical activity part of your daily routine.</p>
            <p>Aim for at least 30 minutes or more of moderate physical activity most days of the week, or at least 150 minutes of moderate physical activity a week. Bouts of activity can be a few minutes during the day. If you haven't been active for a while, start slowly and build up slowly. Also avoid sitting for too long. Try to get up and move if you've been sitting for more than 30 minutes.</p>
          </ul>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'App'
  };
  </script>
  
  <style>
  body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    margin: 0;
    padding: 0;
    background-color: #f4f4f4;
    color: #333;
  }
  .container {
    width: 80%;
    margin: auto;
    overflow: hidden;
  }
  header {
    background: #333;
    color: #fff;
    padding-top: 30px;
    min-height: 70px;
    border-bottom: #007bff 3px solid;
  }
  header a {
    color: #fff;
    text-decoration: none;
    text-transform: uppercase;
    font-size: 16px;
  }
  header ul {
    padding: 0;
    list-style: none;
  }
  header li {
    display: inline;
    padding: 0 20px 0 20px;
  }
  header #branding {
    float: left;
  }
  header #branding h1 {
    margin: 0;
  }
  header nav {
    float: right;
    margin-top: 10px;
  }
  header .highlight, header .current a {
    color: #007bff;
    font-weight: bold;
  }
  header a:hover {
    color: #0056b3;
    font-weight: bold;
  }
  .content {
    text-align: left;
    padding: 20px;
    background: #fff;
    margin-top: 20px;
  }
  .content h2 {
    color: #007bff;
  }
  .content p {
    margin: 5px 0;
  }
  .video-container {
  padding: 20px;
  background: #fff;
  margin-top: 20px;
  text-align: center;
}
video {
  width: 100%;
  max-width: 600px;
  height: auto;
  margin: 0 auto;
}
  </style>