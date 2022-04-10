//select box 자동 선택
function autoSelect(id, val){
	var obj = document.getElementById(id);
	for(var i=0; i < obj.length; i++){
		if(obj.item(i).value == val){
			obj.item(i).selected = true;
		}else{
			obj.item(i).selected = false;	
		}
	}
}

//radio 자동 선택
function autoSelectRadio(name,val) {
	$("input[name="+name+"]").each(function(){
		if(this.value == val) {
			$(this).attr("checked", true);
        }
	});
}

//페이지 이동
function gopage(pageNo) {
	$("#page").val(pageNo);
	document.frm.submit();
}

//해당 글 번호로 이동
function goView(seq){
	$("#seq").val(seq);
	document.frm.submit();
}


function loadSelect(url, destination, params, callBack) {
	var params = $.extend({
		changeFn : '',
		value : 'value', //option의 value에 들어갈 command의 멤버변수명
		text : 'text', //option의 text에 들어갈 command의 멤버변수명
		oneWholeYn : '', //데이타가 한건 일때 전체  옵션 추가 여부
		oneWholeYnTxt : '전체',
		searchViewYn : '', //데이타 최상의에 '선택' 보여주기 여부
		selSelected : '',	//options 출력항목을 넘겨받은 value값으로 selected
		noDataMsg : '', //데이타가 없을 시 출력 할 메세지
		loadingMsg: '조회중입니다..', //로딩중에 출력 할 메세지
		addParam : ''
	}, params);
	var $dest = destination;

	$dest.empty();

	$.ajaxSetup({
        async: false
    });
	
	$.getJSON(url, function (data) {
	    var optionHtml = "";
	    var commandLen = data.length;

	    if (commandLen > 0) {
	        var i = 0;

	        if (params.changeFn != '') {
				$dest.attr("onChange", params.changeFn);
	        }

	        if (params.oneWholeYn != "") {
	            optionHtml += "<option value=''>" + params.oneWholeYnTxt + "</option>";
	        }

	        if (params.searchViewYn != "") {
	            optionHtml += "<option value=''>선택</option>";
	        }

	        $.each(data, function (entryIdx, entry) {
	        	if (params.addParam != ""){
	        		if (params.selSelected == entry[params.value]) {
		                optionHtml += "<option value='" + entry[params.value] + "' data-" + params.addParam + "='" + entry[params.addParam]+"' selected>" + entry[params.text] + "</option>";
		            } else {
		                optionHtml += "<option value='" + entry[params.value] + "' data-" + params.addParam + "='" + entry[params.addParam]+"'>" + entry[params.text] + "</option>";
		            }
	        	}else {
	        		if (params.selSelected == entry[params.value]) {
		                optionHtml += "<option value='" + entry[params.value] + "' selected>" + entry[params.text] + "</option>";
		            } else {
		                optionHtml += "<option value='" + entry[params.value] + "'>" + entry[params.text] + "</option>";
		            }
	        	}
	            i++;
	        });

	    } else {
	        if (params.noDataMsg != '') {
	            optionHtml += "<option value=''>" + params.noDataMsg + "</option>";
	        }
	    }

	    if (callBack) {
	        $.when($dest.append(optionHtml)).then(callBack);
	    } else {
	        $dest.append(optionHtml);
	    }
	});
};