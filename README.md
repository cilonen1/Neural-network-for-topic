На вход будет подаваться контент в виде:			
текст поста, дата, ссылка 			"Джуниор Дос Сантос и Корней Тарасов пытаются завоевать расположение Пэйдж ВанЗант"; 30.01.2020;  https://vk.com/feed?w=wall-197263_4139360
программа берет каждое предложение, прогоняет его через обученную нейросеть 			
на выходе, программа выдает для каждого поста строку длиной n(где n- число тем, по которым мы классифицируем контент) в виде:			
[0.2, 0.01, 0.65,….,0.1]			сумма элементов в каждой строке =1

