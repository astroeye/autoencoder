// 팝업
function pop(url,name,w,h){ 
	window.open(url,name,'width='+w+',height='+h+',scrollbars=no,status=1'); 
}; //Popup(스크롤바없음)
function pops(url,name,w,h){ 
	window.open(url,name,'width='+w+',height='+h+',scrollbars=yes'); 
} //Popup(스크롤바있음)
function newwin(url){ window.open(url,'new','location=1,directories=1,resizable=1,status=1,toolbar=1,menubar=1,scrollbars=1'); 
} //NewPopup(스크롤바없음)


$(document).ready(function() {
	$(".pagination .btn_paging a").hover(function(){
		$(this).each( function(){
			$(this).find('img').attr("src", $(this).find('img').attr("src").replace("_off", "_on")); // 이미지 on
		});
	},function(){
		$(this).each( function(){
			$(this).find('img').attr("src", $(this).find('img').attr("src").replace("_on", "_off")); // 이미지 off
		});
	});
});


// 탭-- 기아T map 고객지원
$(function(){
	/* 탭 */
	$(".tab_content").hide(); 
	$("ul.tabs li:first").addClass("active").show(); 
	$(".tab_content:first").show(); 
	

	$("ul.tabs li").click(function() {
		$("ul.tabs li").removeClass("active"); 
		$(this).addClass("active"); 
		$(".tab_content").hide(); 
		var activeTab = $(this).find("a").attr("href");
		$(activeTab).fadeIn(); 
		return false;
	});
	
})

