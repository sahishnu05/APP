let eyeicon=document.querySelector("#eyeicon")
let password=document.querySelector("#password")
eyeicon.addEventListener("click",()=>{
    if(password.type==="password"){
        password.type="text";
        eyeicon.src="images/eye-open.png";
    }
    else{
        password.type="password";
        eyeicon.src="images/eye-close.png";
    }
})