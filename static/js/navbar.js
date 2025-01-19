const navlink=document.querySelectorAll(".link")
const windowPathname=window.location.pathname;
navlink.forEach(li=>{
    const newLink=new URL(li.href).pathname;
    if(windowPathname===newLink){
      li.classList.add("active")
    }
})