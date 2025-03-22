<template>
<div class="header w">
    <!-- logo模块 -->
    <div class="logo"><a href="#">Healthy Clinic<img src="../assets/clinic.svg" alt=""></a></div>
    <!-- 用户模块 -->
    <div class="user">
        <a href="#"><img src="../assets/user.svg" alt="">
            Log in</a>
    </div>
    <!-- 搜索框 -->
    <div class="search">
        <a href="#"><img src="../assets/search.svg" alt=""></a>
    </div>
</div>
</template>
<style scoped>
.w {
    width: 1200px;
    margin: 0 auto;
    
}

.header {
    
    height: 42px;
    margin: 30px auto;
}
.logo {
    margin-top: -14px;
    float: left;
    width: 40px;
    font-weight: bold;
    font-size: 14px;
}
.logo img {
    width: 40px;
    height: 40px;
    
}
.logo a {
    display: block;
    text-decoration: none;
    color: #000;
}

.user {
    float: left;
    margin-left: 1000px;
    line-height: 50px;
    font-size: 14px;
    font-weight: 500;
    color: #000;
}
.user a{
    color: #000;
}
.user a:hover{
    color: #00a4ff;
}

.search {
    float: right;
    line-height: 50px;
    width: 30px;
}
.search a{
    color: #00a4ff;
}
</style>