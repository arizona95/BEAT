function test(props){
    const {exec} = require("child_process");
    const fs = require('fs')
    var args1 = props.graph
    var args2 = "$"+props.selected
    var args3 = props.state
    var args = args1+"\n"+args2+"\n"+args3
    var cmd = "cd ..&& cd.. && c:\\users\\arizona95\\Anaconda3\\Scripts\\activate tensorflow && python graph_simulator_electron.py"
    fs.writeFile('args.txt', args, err => {
      if (err) {
        console.error(err)
        return
      }
    })
    var process = exec(cmd);
    process.stderr.on("data", function (data) {
      console.error(data.toString());
    });
}