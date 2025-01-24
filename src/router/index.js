import AppLayout from '@/layout/AppLayout.vue';
import FileDoc from '@/views/uikit/FileDoc.vue';
import { createRouter, createWebHistory } from 'vue-router';

const router = createRouter({
    history: createWebHistory(),
    routes: [
        {
            path: '/',
            redirect: '/uikit/file'
        },
        {
            path: '/',
            component: AppLayout,
            children: [
                {
                    path: '/uikit/file',
                    name: 'file',
                    component: FileDoc
                }
            ]
        }
    ]
});

export default router;
