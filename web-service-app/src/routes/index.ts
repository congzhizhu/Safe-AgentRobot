import express from 'express';
import axios from 'axios';
import multer from 'multer';
import FormData from 'form-data';
import fs from 'fs';

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

export function setRoutes(app: express.Express) {
    app.use('/', router);

    router.post('/api/chat', (req, res) => {
        // 说明：对话推理由 Python 侧 Gradio 应用（http://127.0.0.1:7860）承载，
        // 此处不存在可转发的 HTTP 对话接口。早期留空实现会让请求永久挂起，
        // 因此显式返回 501，让前端立即拿到可读错误而不是一直等待。
        res.status(501).json({
            error: '对话接口未启用，请使用 Gradio 主界面 http://127.0.0.1:7860',
            hint: '本服务当前仅提供 /api/upload 文件入库转发能力。'
        });
    });

    router.post('/api/upload', upload.single('file'), async (req, res) => {
        try {
            const { libType } = req.body;
            const file = req.file;

            if (!file) {
                return res.status(400).json({ error: 'No file uploaded' });
            }

            // 转发给 Python 处理 API (8001端口)
            const formData = new FormData();
            formData.append('file', fs.createReadStream(file.path), file.originalname);
            formData.append('lib_type', libType);

            const response = await axios.post('http://127.0.0.1:8001/upload', formData, {
                headers: {
                    ...formData.getHeaders(),
                },
            });

            // 清理本地临时文件
            fs.unlinkSync(file.path);

            res.json(response.data);
        } catch (error: any) {
            console.error("Upload error:", error?.message);
            res.status(500).json({ error: error?.message || "Internal server error" });
        }
    });
}

