# step1: ��������ʹ��tensorflow-gpu����Ȼ����Ҳ����ʹ��python��Ϊ�������񣬺����ٰ�װtensorflow-gpu������
FROM python:3.5

# step2: ����������Ļ���ѧϰ��ص��ļ���������mnist�ļ��У����Ƶ�����ĳ��Ŀ¼�У����磺/home/mnist
COPY ./ /home/ERNIE

# step3 ���������еĹ���Ŀ¼��ֱ���л���/home/mnistĿ¼��
WORKDIR /home/ERNIE

# step4 ��װ����
RUN pip install -r requirements.txt

# step5 ������������ʱ�����������������ֱ������python����
ENTRYPOINT ["sh", "/home/ERNIE/script/zh_task/ernie_base/run_drcd.sh"]
#ENTRYPOINT ["sh", "/home/ERNIE/script/zh_task/ernie_base/run_ChnSentiCorp.sh"]
#ENTRYPOINT ["sh", "/home/ERNIE/script/zh_task/ernie_base/run_drcd_decomp.sh"]