function sendEmail(){
    Email.send({
        Host : "smtp.gmail.com",
        Username : "navam.obeysekara@gmail.com",
        Password : "dnwr whek wrxt tlmq",
        To : 'navam.20211058@iit.ac.lk',
        From : document.getElementById("Email").value,
        Subject : "This is the subject",
        Body : "And this is the body"
    }).then(
      message => alert(message)
    );



    alert("Hello world")
}