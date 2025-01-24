<script setup>
import axios from 'axios';
import { useToast } from 'primevue/usetoast';
import { onMounted, ref } from 'vue';

const toast = useToast();
const uploadedFiles = ref([]); // เก็บรายการไฟล์ที่อัปโหลด

async function onUpload(event) {
    const formData = new FormData();
    for (const file of event.files) {
        formData.append('files', file);
    }

    try {
        // ส่งไฟล์ไปยัง Flask
        const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
        toast.add({ severity: 'success', summary: 'Success', detail: 'Files Uploaded', life: 3000 });
        console.log(response.data);

        // หลังจากอัปโหลดเสร็จ ดึงรายชื่อไฟล์ใหม่
        fetchUploadedFiles();
    } catch (error) {
        toast.add({ severity: 'error', summary: 'Error', detail: 'File Upload Failed', life: 3000 });
        console.error(error);
    }
}

async function fetchUploadedFiles() {
    try {
        const response = await axios.get('http://127.0.0.1:5000/list-files');
        uploadedFiles.value = response.data.files; // เก็บรายการไฟล์ใน uploadedFiles
    } catch (error) {
        console.error('Error fetching files:', error);
    }
}

// ดึงข้อมูลไฟล์เมื่อคอมโพเนนต์ถูกเมาท์
onMounted(() => {
    fetchUploadedFiles();
});
</script>

<template>
    <div class="grid grid-cols-12 gap-8">
        <div class="col-span-full lg:col-span-12">
            <div class="card">
                <div class="font-semibold text-xl mb-4">Upload | CT-Scan Files (.nii.gz)</div>
                <FileUpload name="demo[]" :multiple="false" :maxFileSize="10000000000" customUpload @uploader="onUpload" />
            </div>
            <div class="card">
                <div class="font-semibold text-xl mb-4">Uploaded Files</div>
                <ul>
                    <li v-for="file in uploadedFiles" :key="file" class="mb-2">
                        <a :href="'http://127.0.0.1:5000/uploads/' + file" target="_blank">{{ file }}</a>
                    </li>
                </ul>
                <button @click="fetchUploadedFiles" class="px-4 py-2 border-2 border-blue-400 text-blue-400 rounded hover:bg-blue-400 hover:text-white transition">Refresh Files</button>
            </div>
        </div>
    </div>
</template>
