<template>
    <div id="app">
      
      <div class="container">
        <div class="content">
          <h2>Overview</h2>
          <p>Diabetes mellitus refers to a group of diseases that affect how the body uses blood sugar (glucose). Glucose is an important source of energy for the cells that make up the muscles and tissues. It's also the brain's main source of fuel.</p>
          <p>The main cause of diabetes varies by type. But no matter what type of diabetes you have, it can lead to excess sugar in the blood. Too much sugar in the blood can lead to serious health problems.</p>
          <p>Chronic diabetes conditions include type 1 diabetes and type 2 diabetes. Potentially reversible diabetes conditions include prediabetes and gestational diabetes. Prediabetes happens when blood sugar levels are higher than normal. But the blood sugar levels aren't high enough to be called diabetes. And prediabetes can lead to diabetes unless steps are taken to prevent it. Gestational diabetes happens during pregnancy. But it may go away after the baby is born.</p>
        </div>
        <hr>
        <div class="content">
          <h2>Symptoms</h2>
          <p>Diabetes symptoms depend on how high your blood sugar is. Some people, especially if they have <a href="https://www.mayoclinic.org/diseases-conditions/prediabetes/symptoms-causes/syc-20355278" >prediabetes</a>, 
            <a href="https://www.mayoclinic.org/diseases-conditions/gestational-diabetes/symptoms-causes/syc-20355339">gestational diabetes</a> or <a href="https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/symptoms-causes/syc-20351193">type 2 diabetes</a> , may not have symptoms. 
            In <a href="https://www.mayoclinic.org/diseases-conditions/type-1-diabetes/symptoms-causes/syc-20353011">type 1 diabetes</a> , symptoms tend to come on quickly and be more severe.</p>
          <p>Some of the symptoms of type 1 diabetes and type 2 diabetes are:</p>
          <ul>
            <li>Feeling more thirsty than usual.</li>
            <li>Urinating more often.</li>
            <li>Losing weight without trying.</li>
            <li>Presence of ketones in the urine. Ketones are a byproduct of the breakdown of muscle and fat that happens when there's not enough available insulin.</li>
            <li>Feeling tired and weak.</li>
            <li>Feeling irritable or having other mood changes.</li>
            <li>Having blurry vision.</li>
            <li>Having slow-healing sores.</li>
            <li>Getting a lot of infections, such as gum, skin and vaginal infections.</li>
          </ul>
          <p>Type 1 diabetes can start at any age. But it often starts <a href="https://www.mayoclinic.org/diseases-conditions/type-1-diabetes-in-children/symptoms-causes/syc-20355306">during childhood</a>  
            or teen years. Type 2 diabetes, the more common type, can develop at any age. 
             Type 2 diabetes is more common in people older than 40.
              But <a href="https://www.mayoclinic.org/diseases-conditions/type-2-diabetes-in-children/symptoms-causes/syc-20355318">type 2 diabetes in children</a>  is increasing.</p>
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
    margin: 10px 0;
  }
  </style>