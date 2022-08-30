function test(props){
    console.log(props);
    const exec = require("child_process").exec;
    var process = exec("cd ..&& cd.. && c:\\users\\arizona95\\Anaconda3\\Scripts\\activate tensorflow && python graph_simulator.py");
    process.stderr.on("data", function (data) {
      console.error(data.toString());
    });
}