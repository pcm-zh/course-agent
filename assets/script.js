// ==================== 全局变量 ====================
let selectedFile = null;

// ==================== 1. 会话管理函数 ====================
// === 退出逻辑 ===
function handleExit(event){
    event.preventDefault();
    event.stopPropagation();
    console.log("用户请求退出...");
    
    // 关闭当前标签页
    window.close();
    
    // 注意：window.close() 只能关闭由脚本打开的窗口。
    // 如果是用户手动打开的标签页，现代浏览器可能会忽略此请求。
    // 这是浏览器的安全限制。
}

// === 页面加载初始化逻辑 ===
document.addEventListener("DOMContentLoaded", function(){
    console.log("页面加载完毕，开始初始化...");
    
    // 1. 自动切换到最新会话
    if (window.LATEST_SESSION_NAME && window.LATEST_SESSION_NAME.trim() !== "") {
        console.log("检测到最新会话:", window.LATEST_SESSION_NAME);
        
        // 尝试找到会话列表中的对应按钮
        const sessionButtons = document.querySelectorAll('#session_list button');
        
        let found = false;
        for (let i = 0; i < sessionButtons.length; i++) {
            // 检查按钮文本是否包含最新会话名称
            if (sessionButtons[i].innerText.includes(window.LATEST_SESSION_NAME)) {
                console.log("找到最新会话按钮，准备点击...");
                sessionButtons[i].click();
                found = true;
                break;
            }
        }
        
        if (!found) {
            console.warn("未在列表中找到最新会话按钮，尝试直接调用切换逻辑...");
            // 如果找不到按钮，直接设置隐藏输入框的值并点击触发按钮
            const hiddenInput = document.getElementById('session_name_input');
            const triggerBtn = document.getElementById('trigger_switch_btn');
            if (hiddenInput && triggerBtn) {
                hiddenInput.value = window.LATEST_SESSION_NAME;
                hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
                triggerBtn.click();
            }
        }
    } else {
        console.log("没有最新会话，保持默认状态。");
    }
});

// 切换会话
function triggerSwitchSession(sessionName) {
    console.log('触发会话切换:', sessionName);

    // 1. 找到隐藏的输入框并赋值
    const inputWrapper = document.getElementById('session_name_input');
    if (inputWrapper) {
        const textarea = inputWrapper.querySelector('textarea');
        if (textarea) {
            textarea.value = sessionName;
            textarea.dispatchEvent(new Event('input', { bubbles: true }));
        }
    } else {
        console.error('未找到 session_name_input 元素');
        return;
    }
    
    // 2. 找到触发按钮并点击
    const triggerBtn = document.getElementById('trigger_switch_btn');
    if (triggerBtn) {
        triggerBtn.click();
    } else {
        console.error('未找到 trigger_switch_btn 元素');
    }
}

// 删除会话
function triggerDeleteSession(sessionName) {
    console.log('请求删除会话:', sessionName);
    
    // 1. 弹出确认框
    if (!confirm(`确定要删除会话 "${sessionName}" 吗？此操作无法撤销。`)) {
        return;
    }
    
    // 2. 找到隐藏的输入框并赋值
    const inputWrapper = document.getElementById('session_name_input');
    if (inputWrapper) {
        const textarea = inputWrapper.querySelector('textarea');
        if (textarea) {
            textarea.value = sessionName;
            textarea.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }
    
    // 3. 找到删除触发按钮并点击
    const deleteBtn = document.getElementById('delete_trigger_btn');
    if (deleteBtn) {
        deleteBtn.click();
    }
}

// 新建会话
function triggerNewChat() {
    console.log('触发新建会话');
    
    // 找到新建会话按钮并点击
    const newChatBtn = document.getElementById('new_chat');
    if (newChatBtn) {
        newChatBtn.click();
    } else {
        console.error('未找到 new_chat 元素');
    }
}

// ==================== 2. 文件上传函数 ====================

// 处理文件选择
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        // 保存选中的文件
        selectedFile = file;
        console.log('已选择文件:', file.name);
        
        // 更新输入框显示文件名
        const msgText = document.getElementById('msg_text');
        if (msgText) {
            // 获取输入框内部的textarea元素
            const textarea = msgText.querySelector('textarea');
            if (textarea) {
                textarea.value = file.name;
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
    }
}

// ==================== 3. 发送消息函数 ====================

// 处理发送消息
function handleSend() {
    const msgText = document.getElementById('msg_text');
    if (msgText) {
        // 获取输入框内部的textarea元素
        const textarea = msgText.querySelector('textarea');
        if (textarea && textarea.value.trim() !== '') {
            // 如果有选中的文件，创建文件上传按钮的change事件
            if (selectedFile) {
                const fileUploadInput = document.getElementById('file_upload_btn_real');
                if (fileUploadInput) {
                    // 创建一个新的DataTransfer对象并添加文件
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(selectedFile);
                    
                    // 将文件添加到文件输入框
                    fileUploadInput.files = dataTransfer.files;
                    
                    // 触发change事件
                    fileUploadInput.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
            
            // 触发Gradio的发送事件
            const sendEvent = new Event('submit');
            textarea.dispatchEvent(sendEvent);
            
            // 清空选中的文件
            selectedFile = null;
        }
    }
}

// ==================== 4. 辅助函数 ====================

// 退出处理
function handleExit(event) {
    if (event) event.preventDefault();
    console.log('用户请求退出');
    // window.close(); // 浏览器通常不允许脚本关闭非脚本打开的窗口
    alert("你已成功退出系统"); // 或者重定向到登录页
}

// ==================== 5. 页面初始化与事件绑定 ====================
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM 加载完毕，开始初始化...');

    // 1. 绑定退出按钮
    const exitBtn = document.getElementById('exit_btn');
    if (exitBtn) {
        // 克隆以移除旧的 onclick 属性（如果有的话）
        const newBtn = exitBtn.cloneNode(true);
        exitBtn.parentNode.replaceChild(newBtn, exitBtn);
        newBtn.addEventListener('click', handleExit);
    }

    // 2. 绑定新建会话按钮
    const newChatBtn = document.getElementById('new_chat');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', triggerNewChat);
    }

    // 3. 初始化会话列表
    setTimeout(() => {
        const updateEvent = new Event('update-sessions');
        document.dispatchEvent(updateEvent);
    }, 500);
});

// 监听会话更新事件（用于后续扩展）
document.addEventListener('update-sessions', function() {
    console.log('会话列表需要更新');
});
