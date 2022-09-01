function test(props){
    console.log(props);
    const exec = require("child_process").exec;
    var args1 = props.graph.replaceAll("\"","$")
    var args2 = "$"+props.selected
    var args3 = props.state.replaceAll("\"","$")
    var args = " "+args1+" "+args2+" "+args3
    var cmd = "cd ..&& cd.. && c:\\users\\arizona95\\Anaconda3\\Scripts\\activate tensorflow && python graph_simulator_electron.py"+ args
    console.log(cmd)
    var process = exec(cmd);
    process.stderr.on("data", function (data) {
      console.error(data.toString());
    });
}