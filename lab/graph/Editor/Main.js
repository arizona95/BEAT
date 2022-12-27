const {app, BrowserWindow} = require('electron');
const path = require('path');
const url = require('url');

function createWindow() {
    /*
    * 넓이 1920에 높이 1080의 FHD 풀스크린 앱을 실행시킵니다.
    * */
    const win = new BrowserWindow({
        width:1920,
        height:1080,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });



    const startUrl = "http://localhost:3000/"


    /*
    * startUrl에 배정되는 url을 맨 위에서 생성한 BrowserWindow에서 실행시킵니다.
    * */
    win.loadURL(startUrl);

    win.webContents.openDevTools()

}

app.on('ready', createWindow);
